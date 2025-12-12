import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, GATConv
from typing import List
import logging

logger = logging.getLogger(__name__)


class MultiModalAttention(nn.Module):
    """
    Multi-scale attention mechanism for integrating different feature types
    (e.g., sequence, thermodynamic, positional features) with learnable weights.
    """
    def __init__(self, feature_dims: List[int], hidden_dim: int):
        super().__init__()
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        self.attention_weights = nn.Parameter(torch.ones(len(feature_dims)))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: List[torch.Tensor]):
        projected_features = []
        softmax_weights = torch.softmax(self.attention_weights, dim=0)

        for i, (feat, proj) in enumerate(zip(features, self.feature_projections)):
            projected = proj(feat)
            weighted = projected * softmax_weights[i]
            projected_features.append(weighted)

        combined = torch.stack(projected_features).sum(dim=0)
        return self.layer_norm(combined)


class HierarchicalPooling(nn.Module):
    """
    Enhanced hierarchical attention pooling using multi-head attention
    and residual connections.
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attention_pooling = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, batch):
        if x.size(0) == 0:
            return torch.zeros(1, self.query.size(-1), device=x.device)

        x_reshaped = x.unsqueeze(1)
        batch_size = x_reshaped.size(0)
        query_expanded = self.query.expand(batch_size, -1, -1)

        try:
            attn_output, _ = self.attention_pooling(
                query_expanded, x_reshaped, x_reshaped
            )
            attn_output = attn_output.squeeze(1)
            residual = self.residual_proj(x)
            enhanced_features = self.layer_norm(attn_output + residual)
            pooled = global_mean_pool(enhanced_features, batch)
            return pooled
        except Exception as e:
            logger.warning(f"Hierarchical pooling failed, using fallback: {e}")
            return global_mean_pool(x, batch)


class PositionAwareEncoder(nn.Module):
    """
    Bidirectional encoder with position-aware embeddings for siRNA sequences.
    """
    def __init__(self, input_dim, hidden_dim, max_length=200):
        super().__init__()
        self.forward_lstm = nn.LSTM(input_dim, hidden_dim//2, batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(input_dim, hidden_dim//2, batch_first=True, bidirectional=False)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, positions):
        x_seq = x.unsqueeze(0)
        forward_out, _ = self.forward_lstm(x_seq)
        backward_out, _ = self.backward_lstm(torch.flip(x_seq, [1]))
        backward_out = torch.flip(backward_out, [1])
        bidirectional = torch.cat([forward_out, backward_out], dim=-1).squeeze(0)

        pos_clamped = torch.clamp(positions, 0, self.position_embedding.num_embeddings - 1)
        pos_emb = self.position_embedding(pos_clamped)
        combined = bidirectional + pos_emb
        return self.layer_norm(combined)


class ConvolutionalMotifDetector(nn.Module):
    """
    Multi-scale 1D convolutional layers for capturing RNA motifs of different lengths.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim//4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim//4) for _ in range(4)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            out = F.relu(bn(conv(x)))
            conv_outputs.append(out)
        combined = torch.cat(conv_outputs, dim=1)
        combined = self.dropout(combined)
        return combined.squeeze(0).transpose(0, 1)


class UncertaintyQuantification(nn.Module):
    """
    A simple head to provide uncertainty estimates for predictions.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.variance_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.variance_head(x)
        return mean, torch.exp(log_var)


class OGencoder(nn.Module):
    """
    Main model architecture integrating thermodynamic features with deep learning.

    NEW: Added thermodynamic feature projection and concatenation to graph_repr.
    """
    def __init__(self, foundation_dim=1280, hidden_dim=512, num_heads=8, num_layers=6,
                 dropout=0.15, positional_dim=32, edge_dim=14, num_classes=2):
        super().__init__()

        # Input projection layers
        self.foundation_projection = nn.Sequential(
            nn.Linear(foundation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.onehot_projection = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Custom modules for feature extraction
        self.position_encoder = PositionAwareEncoder(
            input_dim=hidden_dim + hidden_dim // 4,
            hidden_dim=hidden_dim,
            max_length=200
        )
        self.motif_detector = ConvolutionalMotifDetector(
            input_dim=hidden_dim,
            # input_dim=hidden_dim + hidden_dim // 4,
            hidden_dim=hidden_dim
        )
        self.multimodal_attention = MultiModalAttention(
            feature_dims=[hidden_dim, hidden_dim],
            hidden_dim=hidden_dim
        )

        # === NEW: Thermodynamic Feature Projection ===
        self.thermo_projection = nn.Sequential(
            nn.Linear(30, hidden_dim // 4),  # 30 -> 128 for hidden_dim=512
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        # === END NEW SECTION ===

        # Graph convolution layers
        self.transformer_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            self.transformer_convs.append(
                TransformerConv(
                    hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    dropout=dropout, edge_dim=edge_dim, beta=True
                )
            )
            self.gat_convs.append(
                GATConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    concat=True, dropout=dropout, edge_dim=edge_dim
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout + i * 0.02))

        # Advanced pooling mechanism
        self.hierarchical_pool = HierarchicalPooling(hidden_dim, num_heads=num_heads)

        # === UPDATED: Prediction heads with expanded input dimension ===
        # Updated from hidden_dim to (hidden_dim + hidden_dim // 4)
        expanded_dim = hidden_dim + hidden_dim // 4  # 512 + 128 = 640

        self.classification_head = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),  # Increased from 0.2 for overfitting prevention
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.regression_head = UncertaintyQuantification(expanded_dim, 1)

        self.binding_strength_head = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )

        self.stability_head = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        # === END UPDATED SECTION ===

    def forward(self, batch, mode="supervised"):
        # --- 1. Initial Feature Processing ---
        foundation_proj = self.foundation_projection(batch.foundation_features)
        onehot_proj = self.onehot_projection(batch.onehot_features)
        initial_features = torch.cat([foundation_proj, onehot_proj], dim=-1)

        # --- 2. Sequence and Motif Feature Extraction ---
        position_encoded = self.position_encoder(initial_features, batch.positions)
        motif_features = self.motif_detector(position_encoded)
        # motif_features = self.motif_detector(initial_features)
        fused_features = self.multimodal_attention([position_encoded, motif_features])
        x = fused_features

        # --- 3. Graph Representation Learning ---
        for trans_conv, gat_conv, norm, dropout in zip(
            self.transformer_convs, self.gat_convs, self.layer_norms, self.dropout_layers
        ):
            trans_out = trans_conv(x, batch.edge_index, batch.edge_attr)
            gat_out = gat_conv(x, batch.edge_index, batch.edge_attr)
            combined = 0.7 * trans_out + 0.3 * gat_out
            x = norm(x + F.gelu(dropout(combined)))

        # --- 4. Graph Pooling ---
        graph_repr = self.hierarchical_pool(x, batch.batch)

        # === NEW: Integrate Thermodynamic Features ===
        # Check if thermodynamic features are available in the batch
        if hasattr(batch, 'thermodynamic_features'):
            # Aggregate thermodynamic features to graph-level
            # thermodynamic_features shape: [num_nodes, 30]
            thermo_graph = global_mean_pool(batch.thermodynamic_features, batch.batch)

            # Project thermodynamic features
            thermo_projected = self.thermo_projection(thermo_graph)  # [batch_size, 128]

            # Concatenate with graph representation
            graph_repr = torch.cat([graph_repr, thermo_projected], dim=-1)  # [batch_size, 640]
        # === END NEW SECTION ===

        # --- 5. Prediction Heads ---
        outputs = {
            'node_representations': x,
            'graph_representation': graph_repr
        }

        if mode == "supervised":
            outputs['classification'] = self.classification_head(graph_repr)
            reg_mean, reg_var = self.regression_head(graph_repr)
            outputs['regression_mean'] = torch.sigmoid(reg_mean)
            outputs['regression_variance'] = reg_var
            outputs['binding_strength'] = self.binding_strength_head(graph_repr)
            outputs['stability'] = torch.sigmoid(self.stability_head(graph_repr))

        return outputs