#Import necessary libraries
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.stats import pearsonr
import pickle
import os
import logging
from typing import Dict, Any

# Import the core model architecture and data loader utility from ab2.py
try:
    from model import OGencoder
    from train import create_safe_dataloader
except ImportError:
    print("FATAL: 'ab2.py' not found. Please place it in the same directory as this inference script.")
    exit()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Calculates the overlap between the top-k predicted and top-k true siRNAs.
    """
    if len(predictions) < k:
        return 0.0
    top_k_pred_indices = np.argpartition(predictions, -k)[-k:]
    top_k_true_indices = np.argpartition(targets, -k)[-k:]
    overlap = len(set(top_k_pred_indices) & set(top_k_true_indices))
    return overlap / k


def display_checkpoint_info(checkpoint: Dict[str, Any]):
    """
    Extracts and displays metadata from the model checkpoint.
    """
    logger.info("=" * 80)
    logger.info("CHECKPOINT INFORMATION")
    logger.info("=" * 80)
    
    # Display basic checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"  Checkpoint Epoch:     {checkpoint['epoch']}")
    
    if 'timestamp' in checkpoint:
        logger.info(f"  Saved at:             {checkpoint['timestamp']}")
    
    # Display training metrics if available
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        logger.info("\n--- Training Metrics (from checkpoint) ---")
        
        if 'train' in metrics:
            train_metrics = metrics['train']
            logger.info("  Training Set:")
            for key, value in train_metrics.items():
                logger.info(f"    {key:<20}: {value:.6f}")
        
        if 'val' in metrics:
            val_metrics = metrics['val']
            logger.info("\n  Validation Set:")
            for key, value in val_metrics.items():
                logger.info(f"    {key:<20}: {value:.6f}")
    else:
        logger.warning("  No metrics found in checkpoint.")
    
    logger.info("=" * 80)


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, Any]:
    """
    Runs the model on the provided data and computes a comprehensive set of metrics.
    """
    model.eval()
    all_class_preds, all_class_targets = [], []
    all_reg_preds, all_reg_vars, all_reg_targets = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch, mode="supervised")
            
            all_class_preds.append(outputs['classification'].detach())
            all_class_targets.append(batch.y_class.detach())
            all_reg_preds.append(outputs['regression_mean'].detach())
            all_reg_vars.append(outputs['regression_variance'].detach())
            all_reg_targets.append(batch.y.detach())

    # Concatenate results from all batches
    class_probs = torch.softmax(torch.cat(all_class_preds), dim=1).cpu().numpy()
    class_pred_labels = np.argmax(class_probs, axis=1)
    class_true_labels = torch.cat(all_class_targets).cpu().numpy()
    
    reg_preds_np = torch.cat(all_reg_preds).squeeze().cpu().numpy()
    reg_targets_np = torch.cat(all_reg_targets).cpu().numpy()
    reg_vars_np = torch.cat(all_reg_vars).squeeze().cpu().numpy()

    # --- Compute Classification Metrics ---
    accuracy = accuracy_score(class_true_labels, class_pred_labels)
    f1 = f1_score(class_true_labels, class_pred_labels, average='weighted', zero_division=0)
    auc_roc, auc_pr = 0.5, 0.5
    if len(np.unique(class_true_labels)) > 1:
        auc_roc = roc_auc_score(class_true_labels, class_probs[:, 1])
        precision, recall, _ = precision_recall_curve(class_true_labels, class_probs[:, 1])
        auc_pr = auc(recall, precision)

    # --- Compute Regression Metrics ---
    finite_mask = np.isfinite(reg_preds_np) & np.isfinite(reg_targets_np)
    if np.sum(~finite_mask) > 0:
        logger.warning(f"Filtered {np.sum(~finite_mask)} NaN/inf values from regression results.")
    
    reg_preds_clean = reg_preds_np[finite_mask]
    reg_targets_clean = reg_targets_np[finite_mask]
    
    pcc, _ = pearsonr(reg_preds_clean, reg_targets_clean) if len(reg_preds_clean) > 1 else (0, 0)
    mse = np.mean((reg_targets_clean - reg_preds_clean)**2) if len(reg_targets_clean) > 0 else 0
    top_10_acc = calculate_top_k_accuracy(reg_preds_clean, reg_targets_clean, k=10)

    return {
        'metrics': {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'pcc': pcc if not np.isnan(pcc) else 0,
            'mse': mse,
            'top_10_accuracy': top_10_acc,
            'mean_uncertainty': reg_vars_np.mean()
        },
        'outputs': {
            'regression_predictions': reg_preds_np,
            'regression_targets': reg_targets_np,
            'classification_probabilities': class_probs,
            'classification_targets': class_true_labels
        }
    }


def compare_metrics(checkpoint_metrics: Dict[str, Any], inference_metrics: Dict[str, float]):
    """
    Compares checkpoint metrics with current inference metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("METRICS COMPARISON: Checkpoint vs Current Test")
    logger.info("=" * 80)
    
    if 'val' in checkpoint_metrics:
        val_metrics = checkpoint_metrics['val']
        
        logger.info(f"{'Metric':<25} {'Checkpoint':<15} {'Current':<15} {'Difference':<15}")
        logger.info("-" * 80)
        
        for key in inference_metrics.keys():
            if key in val_metrics:
                checkpoint_val = val_metrics[key]
                current_val = inference_metrics[key]
                diff = current_val - checkpoint_val
                diff_str = f"{diff:+.6f}"
                logger.info(f"{key:<25} {checkpoint_val:<15.6f} {current_val:<15.6f} {diff_str:<15}")
        
        logger.info("=" * 80)
    else:
        logger.warning("No validation metrics in checkpoint to compare.")


def main():
    """
    Main function to orchestrate the inference pipeline.
    """
    logger.info("=" * 80)
    logger.info("OligoGraph Test Script with Checkpoint Metrics")
    logger.info("=" * 80)

    # --- Configuration ---
    MODEL_PATH = 'Checkpoints/best_model.pt'
    DATA_PATH = 'Data/processed_data/val_data.pkl'
    OUTPUT_CSV_PATH = 'Data/test_results.csv'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Prerequisite Checks ---
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model checkpoint not found at '{MODEL_PATH}'.")
        logger.error("Please ensure the model file exists.")
        return
    if not os.path.exists(DATA_PATH):
        logger.error(f"Processed data not found at '{DATA_PATH}'.")
        logger.error("Please run the preprocessing script first to generate the data file.")
        return

    logger.info(f"Using device: {DEVICE}")

    # --- Model Loading ---
    try:
        logger.info(f"Loading model checkpoint from {MODEL_PATH}...")
        # checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        # Display checkpoint information
        #display_checkpoint_info(checkpoint)
        
        # Initialize model
        model = OGencoder(
            foundation_dim=1280,
            hidden_dim=512,
            num_heads=8,
            num_layers=8,
            dropout=0.15
        ).to(DEVICE)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("\nModel loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load the model. Error: {e}")
        return

    # --- Data Loading ---
    try:
        logger.info(f"\nLoading test data from {DATA_PATH}...")
        with open(DATA_PATH, 'rb') as f:
            test_data = pickle.load(f)
        
        test_loader = create_safe_dataloader(test_data, batch_size=64, shuffle=False)
        if not test_loader:
            logger.error("Failed to create test data loader.")
            return
        logger.info(f"Test data loaded: {len(test_data)} samples.")
    except Exception as e:
        logger.error(f"Failed to load or process test data. Error: {e}")
        return

    # --- Run Inference and Evaluation ---
    logger.info("\nRunning testing on the test set...")
    results = evaluate_model(model, test_loader, DEVICE)
    metrics = results['metrics']
    outputs = results['outputs']
    logger.info("Test complete.")

    # --- Display Current Inference Metrics ---
    logger.info("\n" + "=" * 80)
    logger.info("CURRENT TEST METRICS")
    logger.info("=" * 80)
    logger.info(f"  Accuracy:             {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score:             {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:              {metrics['auc_roc']:4f}")
    logger.info(f"  AUC-PR:               {metrics['auc_pr']:.4f}")
    logger.info(f"  PCC (Pearson):        {metrics['pcc']:.4f}")
    logger.info(f"  MSE (Regression):     {metrics['mse']:.4f}")
    #logger.info(f"  Top-10 Accuracy:      {metrics['top_10_accuracy']:.6f}")
    #logger.info(f"  Mean Uncertainty:     {metrics['mean_uncertainty']:.6f}")
    logger.info("=" * 80)

    # --- Compare with Checkpoint Metrics ---
    # if 'metrics' in checkpoint:
    #     compare_metrics(checkpoint['metrics'], metrics)

    # --- Save Results to CSV ---
    try:
        logger.info(f"\nSaving prediction results to {OUTPUT_CSV_PATH}...")
        results_df = pd.DataFrame({
            'regression_target': outputs['regression_targets'],
            'regression_prediction': outputs['regression_predictions'],
            'classification_target': outputs['classification_targets'],
            'probability_class_0': outputs['classification_probabilities'][:, 0],
            'probability_class_1': outputs['classification_probabilities'][:, 1]
        })
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Could not save results to CSV. Error: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Test script finished.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


