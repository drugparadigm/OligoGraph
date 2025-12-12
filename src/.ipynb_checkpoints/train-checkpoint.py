import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.stats import pearsonr
import pickle
import numpy as np
import time
from datetime import datetime
import logging
import os
import random
from typing import Dict

from model import OGencoder

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in efficacy prediction.
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class WarmupCosineScheduler:
    """
    Enhanced learning rate scheduler with a warmup phase followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs=10, total_epochs=150, base_lr=1e-4, min_lr=1e-6, restart_period=50):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.restart_period = restart_period
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        cycle_epoch = self.current_epoch % self.restart_period

        if cycle_epoch < self.warmup_epochs:
            lr = self.base_lr * (cycle_epoch / self.warmup_epochs)
        else:
            progress = (cycle_epoch - self.warmup_epochs) / (self.restart_period - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class AdvancedDataAugmentation:
    """
    Comprehensive data augmentation strategies for RNA sequences and features.
    """
    def __init__(self, device):
        self.device = device

    def sequence_mutation(self, onehot_features, mutation_rate=0.05):
        mutated = onehot_features.clone()
        seq_len = mutated.size(0)
        num_mutations = max(1, int(seq_len * mutation_rate))
        conservative_mutations = {0: [0, 3], 1: [1, 2], 2: [1, 2], 3: [0, 3]}

        for _ in range(num_mutations):
            pos = torch.randint(0, seq_len, (1,)).item()
            original_base = torch.argmax(mutated[pos]).item()
            new_base = random.choice(conservative_mutations[original_base])
            mutated[pos] = F.one_hot(torch.tensor(new_base), num_classes=4).float()
        return mutated

    def feature_noise_injection(self, features, noise_std=0.1):
        noise = torch.randn_like(features) * noise_std
        return features + noise

    def thermodynamic_perturbation(self, edge_attr, perturbation_ratio=0.1):
        if edge_attr.numel() == 0: 
            return edge_attr
        perturbed = edge_attr.clone()
        num_edges_to_perturb = max(1, int(perturbed.size(0) * perturbation_ratio))
        indices = torch.randperm(perturbed.size(0))[:num_edges_to_perturb]
        noise = torch.randn_like(perturbed[indices]) * 0.05
        perturbed[indices] += noise
        return perturbed


class EnhancedTrainingStrategy:
    """
    Wrapper class for the entire training process.
    """
    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config or {}

        param_groups = [
            {'params': model.transformer_convs.parameters(), 'lr': 1e-4},
            {'params': model.classification_head.parameters(), 'lr': 2e-4},
            {'params': model.regression_head.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_epochs=10, total_epochs=150
        )

        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.augmentation = AdvancedDataAugmentation(device)

        self.patience = 30
        self.best_score = -float('inf')
        self.patience_counter = 0

    def compute_advanced_loss(self, outputs, batch):
        """Computes multi-component loss with adaptive weighting."""
        losses = {}

        # Classification loss
        if 'classification' in outputs:
            losses['classification'] = self.focal_loss(outputs['classification'], batch.y_class)

        # Regression loss with uncertainty
        if 'regression_mean' in outputs and 'regression_variance' in outputs:
            mean_pred = outputs['regression_mean'].squeeze()
            var_pred = torch.clamp(outputs['regression_variance'].squeeze(), min=1e-4)
            targets = batch.y

            nll_loss = 0.5 * (torch.log(var_pred) + ((targets - mean_pred)**2 / var_pred))
            variance_penalty = 0.01 * torch.mean(-torch.log(var_pred))
            losses['regression'] = torch.mean(nll_loss) + variance_penalty

        # Ranking loss
        if len(batch.y) > 1:
            n = len(batch.y)
            ranking_losses = []
            for i in range(min(n, 10)):
                for j in range(i+1, min(n, 10)):
                    if abs(batch.y[i] - batch.y[j]) > 0.1:
                        pred_i, pred_j = outputs['regression_mean'][i], outputs['regression_mean'][j]
                        target_sign = torch.sign(batch.y[i] - batch.y[j])
                        ranking_losses.append(
                            self.ranking_loss(pred_i, pred_j, target_sign.view_as(pred_i))
                        )
            if ranking_losses:
                losses['ranking'] = torch.stack(ranking_losses).mean()

        # Auxiliary task losses
        if 'binding_strength' in outputs and hasattr(batch, 'binding_strength_targets'):
            losses['binding'] = F.cross_entropy(
                outputs['binding_strength'], batch.binding_strength_targets
            )

        if 'stability' in outputs and hasattr(batch, 'stability_targets'):
            losses['stability'] = self.smooth_l1_loss(
                outputs['stability'].squeeze(), batch.stability_targets
            )

        # Calculate total loss
        valid_loss_components = []
        weights = {
            'classification': 2.0,
            'regression': 1.5,
            'ranking': 0.3,
            'binding': 0.5,
            'stability': 0.2
        }

        for name, value in losses.items():
            if value is not None and not torch.isnan(value):
                valid_loss_components.append(weights.get(name, 1.0) * value)

        if not valid_loss_components:
            return None, losses

        total_loss = torch.stack(valid_loss_components).sum()
        return total_loss, losses

    def train_epoch_with_augmentation(self, dataloader):
        """Runs a single training epoch with data augmentation."""
        self.model.train()
        total_loss = 0
        all_predictions, all_targets = [], []

        for batch in dataloader:
            batch = batch.to(self.device)

            # Apply augmentations
            if random.random() < 0.7:
                if random.random() < 0.3:
                    batch.onehot_features = self.augmentation.sequence_mutation(
                        batch.onehot_features
                    )
                if random.random() < 0.4:
                    batch.foundation_features = self.augmentation.feature_noise_injection(
                        batch.foundation_features
                    )
                if random.random() < 0.3:
                    batch.edge_attr = self.augmentation.thermodynamic_perturbation(
                        batch.edge_attr
                    )

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(batch, mode="supervised")

            loss, _ = self.compute_advanced_loss(outputs, batch)

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.append(outputs['regression_mean'].detach())
            all_targets.append(batch.y.detach())

        if not all_predictions:
            logger.warning("All training batches were skipped due to numerical instability.")
            return {'loss': float('nan'), 'pcc': 0}

        all_preds = torch.cat(all_predictions).cpu().numpy().flatten()
        all_targs = torch.cat(all_targets).cpu().numpy().flatten()
        pcc, _ = pearsonr(all_targs, all_preds) if len(all_preds) > 1 else (0, 0)

        return {'loss': total_loss / len(dataloader), 'pcc': pcc if not np.isnan(pcc) else 0}

    def validate_with_uncertainty(self, dataloader):
        """Runs validation loop and computes comprehensive metrics."""
        self.model.eval()
        all_class_preds, all_class_targets = [], []
        all_reg_preds, all_reg_vars, all_reg_targets = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)

                outputs = self.model(batch, mode="supervised")

                all_class_preds.append(outputs['classification'].detach())
                all_class_targets.append(batch.y_class.detach())
                all_reg_preds.append(outputs['regression_mean'].detach())
                all_reg_vars.append(outputs['regression_variance'].detach())
                all_reg_targets.append(batch.y.detach())

        # Classification metrics
        class_probs = torch.softmax(torch.cat(all_class_preds), dim=1).cpu().numpy()
        class_pred_labels = np.argmax(class_probs, axis=1)
        class_true_labels = torch.cat(all_class_targets).cpu().numpy()

        accuracy = accuracy_score(class_true_labels, class_pred_labels)
        f1 = f1_score(class_true_labels, class_pred_labels, average='weighted', zero_division=0)

        auc_roc, auc_pr = 0.5, 0.5
        if len(np.unique(class_true_labels)) > 1:
            auc_roc = roc_auc_score(class_true_labels, class_probs[:, 1])
            precision, recall, _ = precision_recall_curve(class_true_labels, class_probs[:, 1])
            auc_pr = auc(recall, precision)

        # Regression metrics
        reg_preds_np = torch.cat(all_reg_preds).squeeze().cpu().numpy()
        reg_targets_np = torch.cat(all_reg_targets).cpu().numpy()

        finite_mask = np.isfinite(reg_preds_np) & np.isfinite(reg_targets_np)
        if np.sum(~finite_mask) > 0:
            logger.warning(
                f"Found and filtered {np.sum(~finite_mask)} NaN/inf values in predictions/targets."
            )

        reg_preds_np_clean = reg_preds_np[finite_mask]
        reg_targets_np_clean = reg_targets_np[finite_mask]

        pcc, _ = pearsonr(reg_preds_np_clean, reg_targets_np_clean) if len(reg_preds_np_clean) > 1 else (0, 0)
        mse = np.mean((reg_targets_np_clean - reg_preds_np_clean)**2) if len(reg_preds_np_clean) > 0 else 0
        top_k_acc = self.calculate_top_k_accuracy(reg_preds_np_clean, reg_targets_np_clean, k=10)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'pcc': pcc if not np.isnan(pcc) else 0,
            'mse': mse,
            'top_k_accuracy': top_k_acc,
            'uncertainty': torch.cat(all_reg_vars).mean().item()
        }

    def calculate_top_k_accuracy(self, predictions, targets, k=10):
        """Calculates overlap between top-k predicted and top-k true siRNAs."""
        if len(predictions) < k:
            return 0.0
        top_k_pred_indices = np.argpartition(predictions, -k)[-k:]
        top_k_true_indices = np.argpartition(targets, -k)[-k:]
        overlap = len(set(top_k_pred_indices) & set(top_k_true_indices))
        return overlap / k

    def should_stop_early(self, val_score):
        """Checks if training should stop based on validation performance."""
        if val_score > self.best_score:
            self.best_score = val_score
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience


# === Utility Functions ===

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def create_safe_dataloader(data_list, batch_size=32, shuffle=True):
    """Creates a PyTorch Geometric DataLoader with error handling."""
    valid_data = [d for d in data_list if hasattr(d, 'x') and d.x is not None]
    if len(valid_data) < len(data_list):
        logger.warning(f"Filtered out {len(data_list) - len(valid_data)} invalid samples.")
    if not valid_data:
        return None
    return DataLoader(
        valid_data, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True
    )


def save_model_with_metadata(model, optimizer, epoch, metrics, filepath):
    """Saves model checkpoint with training metadata."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }, filepath)


# === Main Training Pipeline ===

def main():
    """Main function to orchestrate the entire training pipeline."""
    logger.info("=" * 30)
    logger.info("Started Training")
    logger.info("=" * 30)

    # Setup
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = OGencoder(
        foundation_dim=1280,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        dropout=0.15
    ).to(device)

    # Load pre-trained weights (if available)
    pretrained_path = ""
    # pretrained_path = "enhanced_checkpoints_8257roc_615pcc/pretrained_backbone_epoch_60.pt"
    # pretrained_path = "enhanced_checkpoints/pretrained_backbone_final.pt"
    if os.path.exists(pretrained_path):
        try:
            logger.info(f"Loading pre-trained model weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            logger.info("Successfully loaded pre-trained weights.")
        except Exception as e:
            logger.error(f"Could not load pre-trained weights: {e}. Starting from scratch.")
    else:
        logger.warning("No pre-trained model found. Starting from scratch.")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = "Checkpoints/training_checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data loading
    try:
        with open('Data/processed_data/train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('Data/processed_data/val_data.pkl', 'rb') as f:
            val_data = pickle.load(f)
    except FileNotFoundError:
        logger.error("Processed data not found. Please run preprocessing script first.")
        return

    train_loader = create_safe_dataloader(train_data, batch_size=64, shuffle=True)
    val_loader = create_safe_dataloader(val_data, batch_size=64, shuffle=False)

    if not train_loader or not val_loader:
        logger.error("Failed to create dataloaders, exiting.")
        return

    # Training initialization
    trainer = EnhancedTrainingStrategy(model, device)
    best_score = -float('inf')
    start_time = time.time()

    logger.info("=" * 120)
    header = (f"{'Time':<10} {'Epoch':<6} {'Train Loss':<12} {'Train PCC':<10} "
              f"{'Val Acc':<10} {'Val AUC-ROC':<12} {'Val AUC-PR':<11} {'Val PCC':<10} "
              f"{'Top-10':<8} {'LR':<10}")
    logger.info(header)
    logger.info("=" * 120)

    # Main training loop
    for epoch in range(150):
        train_metrics = trainer.train_epoch_with_augmentation(train_loader)
        val_metrics = trainer.validate_with_uncertainty(val_loader)
        lr = trainer.scheduler.step()

        val_score = (0.6 * val_metrics['auc_roc'] + 0.4 * val_metrics['pcc'])

        if not np.isnan(val_score) and val_score > best_score:
            best_score = val_score
            logger.info(f"🎯 NEW BEST MODEL at epoch {epoch+1} with score: {val_score:.4f}")
            save_path = os.path.join(checkpoint_dir, 'best_enhanced_model.pt')
            save_model_with_metadata(
                model, trainer.optimizer, epoch + 1,
                {'train': train_metrics, 'val': val_metrics}, save_path
            )

        log_line = (f"{format_time(time.time() - start_time):<10} {epoch+1:<6} "
                    f"{train_metrics['loss']:<12.4f} {train_metrics['pcc']:<10.4f} "
                    f"{val_metrics['accuracy']:<10.4f} {val_metrics['auc_roc']:<12.4f} "
                    f"{val_metrics['auc_pr']:<11.4f} {val_metrics['pcc']:<10.4f} "
                    f"{val_metrics['top_k_accuracy']:<8.4f} {lr:<10.6f}")
        logger.info(log_line)

        if trainer.should_stop_early(val_score):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_model_with_metadata(
                model, trainer.optimizer, epoch + 1,
                {'train': train_metrics, 'val': val_metrics}, checkpoint_path
            )

    logger.info(" TRAINING COMPLETED! ")
    logger.info(f"Best validation score achieved: {best_score:.4f}")


if __name__ == "__main__":
    main()