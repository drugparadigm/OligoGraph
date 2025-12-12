# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import global_mean_pool
import pickle
import os
import numpy as np
import logging
import time

# Import the core model architecture and utilities from training script
from train import (
    OGencoder, 
    format_time, 
    create_safe_dataloader, 
    save_model_with_metadata
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Self-Supervised Learning Modules ===

class GraphAugmentation:
    """
    Applies augmentations to a graph to create two correlated 'views' for contrastive learning.
    """
    def __init__(self, node_mask_rate=0.15, edge_drop_rate=0.2):
        self.node_mask_rate = node_mask_rate
        self.edge_drop_rate = edge_drop_rate

    def __call__(self, data):
        # Create two augmented views of the same data
        data_view1 = self.augment(data.clone())
        data_view2 = self.augment(data.clone())
        return data_view1, data_view2

    def augment(self, data):
        num_nodes = data.num_nodes
        
        # 1. Mask Node Features (specifically the one-hot encoding)
        mask = torch.rand(num_nodes) < self.node_mask_rate
        data.masked_indices = mask.nonzero(as_tuple=False).view(-1)
        data.original_onehot = data.onehot_features.clone()
        # Use a simple zero vector as mask token
        data.onehot_features[data.masked_indices] = 0

        # 2. Drop Edges
        edge_index, edge_attr = dropout_adj(
            data.edge_index, data.edge_attr, p=self.edge_drop_rate, training=True
        )
        data.edge_index, data.edge_attr = edge_index, edge_attr
        
        return data


class PretrainableOligoGraph(nn.Module):
    """
    A wrapper around the main model that adds heads for self-supervised tasks.
    """
    def __init__(self, foundation_dim=1280, hidden_dim=512, **kwargs):
        super().__init__()
        # The core GNN backbone - using the model from train.py
        self.backbone = OGencoder(foundation_dim, hidden_dim, **kwargs)

        # Head for Masked Node Prediction (predicting the nucleotide type)
        self.node_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 classes: A, U, G, C
        )

        # The backbone now outputs a graph representation that includes thermodynamic features.
        # The dimension is expanded_dim = hidden_dim + hidden_dim // 4
        expanded_dim = hidden_dim + hidden_dim // 4

        # Projection head for Contrastive Learning
        self.graph_projection_head = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, batch, mode="pretrain"):
        # Get the final node and graph representations from the backbone model
        outputs = self.backbone(batch, mode="supervised")
        
        node_repr = outputs['node_representations']
        graph_repr = outputs['graph_representation']

        if mode == "pretrain":
            # Pass representations through the SSL heads
            masked_node_preds = self.node_prediction_head(node_repr)
            graph_projection = self.graph_projection_head(graph_repr)
            return masked_node_preds, graph_projection
        
        return node_repr, graph_repr


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, p=2, dim=1)

        similarity_matrix = torch.mm(z, z.T) / self.temperature
        
        # Create labels for positive pairs
        mask = torch.eye(batch_size, device=z.device).bool()
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size), 
            torch.diag(similarity_matrix, -batch_size)
        ])

        # Select negative samples (all except the positive pair and self-comparison)
        negatives_mask = ~torch.eye(2 * batch_size, device=z.device).bool()
        negatives = similarity_matrix[negatives_mask].view(2 * batch_size, -1)

        logits = torch.cat([positives.view(-1, 1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)

        loss = self.criterion(logits, labels) / (2 * batch_size)
        return loss


class PretrainingStrategy:
    """
    Encapsulates the self-supervised pre-training loop.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.contrastive_loss_fn = NTXentLoss(temperature=0.1)
        self.node_pred_loss_fn = nn.CrossEntropyLoss()
        self.augmentation = GraphAugmentation()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, total_contrast_loss, total_node_loss = 0, 0, 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Create two augmented views for contrastive learning
            view1, view2 = self.augmentation(batch.to(self.device))

            # Get predictions for both views
            masked_node_preds1, graph_proj1 = self.model(view1)
            masked_node_preds2, graph_proj2 = self.model(view2)

            # --- 1. Contrastive Loss ---
            loss_contrast = self.contrastive_loss_fn(graph_proj1, graph_proj2)

            # --- 2. Masked Node Prediction Loss ---
            # Get original nucleotide labels for the masked positions
            true_labels1 = torch.argmax(view1.original_onehot[view1.masked_indices], dim=1)
            pred_logits1 = masked_node_preds1[view1.masked_indices]
            
            true_labels2 = torch.argmax(view2.original_onehot[view2.masked_indices], dim=1)
            pred_logits2 = masked_node_preds2[view2.masked_indices]

            loss_node1 = self.node_pred_loss_fn(pred_logits1, true_labels1) if len(true_labels1) > 0 else 0
            loss_node2 = self.node_pred_loss_fn(pred_logits2, true_labels2) if len(true_labels2) > 0 else 0
            loss_node = (loss_node1 + loss_node2) / 2

            # --- 3. Combined Loss ---
            # Weight the two losses
            loss = loss_contrast + 0.5 * loss_node
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_contrast_loss += loss_contrast.item()
            total_node_loss += loss_node.item() if isinstance(loss_node, torch.Tensor) else loss_node

        self.scheduler.step()
        
        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'contrast_loss': total_contrast_loss / num_batches,
            'node_loss': total_node_loss / num_batches
        }


# === Main Pre-training Pipeline ===
def main_pretrain():
    logger.info("=" * 30)
    logger.info("-----Pretraing Started-----")
    logger.info("=" * 30)

    # --- Setup ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Use the new pre-trainable model wrapper
    model = PretrainableOligoGraph(
        foundation_dim=1280, 
        hidden_dim=512, 
        num_heads=8, 
        num_layers=8, 
        dropout=0.15
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = "Checkpoints/pretrain_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Data Loading ---
    # NOTE: For pre-training, you should ideally use a much larger UNLABELED dataset.
    # Here, we try to load pretrain_data.pkl, or fall back to train_data.pkl
    try:
        with open('Data/processed_data/pretrain_data.pkl', 'rb') as f:
            pretrain_data = pickle.load(f)
        logger.info("Loaded pretrain_data.pkl for pre-training")
    except FileNotFoundError:
        logger.warning("`pretrain_data.pkl` not found. Reusing `train_data.pkl` for demonstration.")
        try:
            with open('Data/processed_data/train_data.pkl', 'rb') as f:
                pretrain_data = pickle.load(f)
        except FileNotFoundError:
            logger.error("No training data found. Please run preprocessing first.")
            return

    pretrain_loader = create_safe_dataloader(pretrain_data, batch_size=128, shuffle=True)
    if not pretrain_loader:
        logger.error("Failed to create pre-training dataloader, exiting.")
        return

    # --- Pre-training Initialization ---
    trainer = PretrainingStrategy(model, device)
    start_time = time.time()
    num_epochs = 100  # Pre-training usually requires more epochs

    logger.info("=" * 100)
    header = f"{'Time':<10} {'Epoch':<6} {'Total Loss':<12} {'Contrast Loss':<15} {'Node Pred Loss':<18} {'LR':<10}"
    logger.info(header)
    logger.info("=" * 100)

    # --- Main Pre-training Loop ---
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(pretrain_loader)
        lr = trainer.scheduler.get_last_lr()[0]

        log_line = (f"{format_time(time.time() - start_time):<10} {epoch+1:<6} "
                    f"{train_metrics['loss']:<12.4f} {train_metrics['contrast_loss']:<15.4f} "
                    f"{train_metrics['node_loss']:<18.4f} {lr:<10.6f}")
        logger.info(log_line)

        # Save the pre-trained backbone model periodically
        if (epoch + 1) % 20 == 0:
            logger.info(f"Saving pre-trained backbone at epoch {epoch+1}")
            save_path = os.path.join(checkpoint_dir, f'pretrained_backbone_epoch_{epoch+1}.pt')
            # IMPORTANT: We only save the state dict of the backbone GNN
            torch.save(model.backbone.state_dict(), save_path)

    logger.info("🚀 PRE-TRAINING COMPLETED! 🚀")
    logger.info("Saving final pre-trained model backbone.")
    final_save_path = os.path.join(checkpoint_dir, 'pretrained_backbone_final.pt')
    torch.save(model.backbone.state_dict(), final_save_path)
    logger.info(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main_pretrain()