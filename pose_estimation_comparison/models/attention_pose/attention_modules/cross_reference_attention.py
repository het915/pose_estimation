"""Cross-reference attention for enhanced multi-view matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossReferenceAttention(nn.Module):
    """
    Cross-reference attention for multi-view matching.

    Key contribution: Leverages FoundationPose's multi-view references
    through explicit attention mechanism rather than simple cosine similarity.

    This allows the model to selectively attend to the most relevant
    reference views even under heavy occlusion.
    """

    def __init__(
        self,
        query_dim: int = 256,
        key_dim: int = 256,
        num_heads: int = 8,
        num_reference_views: int = 8,
        dropout: float = 0.1,
        attention_pooling: str = "max",  # max, mean, or learned
    ):
        """
        Args:
            query_dim: Query feature dimension
            key_dim: Key feature dimension
            num_heads: Number of attention heads
            num_reference_views: Number of reference views
            dropout: Dropout rate
            attention_pooling: How to pool multi-view attention
        """
        super(CrossReferenceAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.num_reference_views = num_reference_views
        self.attention_pooling = attention_pooling

        assert query_dim % num_heads == 0

        # Query projection (from fused query features)
        self.query_proj = nn.Conv2d(query_dim, query_dim, 1)

        # Key and Value projections (from reference features)
        self.key_proj = nn.Conv2d(key_dim, query_dim, 1)
        self.value_proj = nn.Conv2d(key_dim, query_dim, 1)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 1),
            nn.BatchNorm2d(query_dim),
            nn.ReLU(),
        )

        # View importance weighting
        self.view_importance = nn.Sequential(
            nn.Linear(query_dim, num_reference_views),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout)

        # Learned pooling (if selected)
        if attention_pooling == "learned":
            self.pooling_weights = nn.Parameter(torch.ones(num_reference_views) / num_reference_views)

    def forward(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-reference attention.

        Args:
            query_features: (B, C, H, W) query features
            reference_features: (B, num_ref, C, H, W) reference features

        Returns:
            enhanced_query: (B, C, H, W) attention-enhanced query features
            matching_scores: (B, num_ref) view matching scores
            attention_map: (B, num_ref, H, W) spatial attention per view
        """
        B, C, H, W = query_features.shape
        num_ref = reference_features.shape[1]

        # Project query
        Q = self.query_proj(query_features)  # (B, C, H, W)

        # Process each reference view
        attended_features_list = []
        attention_maps_list = []

        for i in range(num_ref):
            ref_feat = reference_features[:, i]  # (B, C, H, W)

            # Project key and value
            K = self.key_proj(ref_feat)
            V = self.value_proj(ref_feat)

            # Multi-head cross-attention
            attended, attn_map = self._cross_attention(Q, K, V)  # (B, C, H, W), (B, H, W)

            attended_features_list.append(attended)
            attention_maps_list.append(attn_map)

        # Stack attended features from all views
        attended_features = torch.stack(attended_features_list, dim=1)  # (B, num_ref, C, H, W)
        attention_maps = torch.stack(attention_maps_list, dim=1)  # (B, num_ref, H, W)

        # Compute view importance scores
        query_global = F.adaptive_avg_pool2d(query_features, 1).view(B, C)  # (B, C)
        view_scores = self.view_importance(query_global)  # (B, num_ref)

        # Pool attended features across views
        if self.attention_pooling == "max":
            enhanced_query, _ = torch.max(attended_features, dim=1)  # (B, C, H, W)
        elif self.attention_pooling == "mean":
            enhanced_query = attended_features.mean(dim=1)
        elif self.attention_pooling == "learned":
            # Weighted combination using learned weights
            weights = F.softmax(self.pooling_weights, dim=0).view(1, num_ref, 1, 1, 1)
            enhanced_query = (attended_features * weights).sum(dim=1)
        else:
            # Weighted by view importance
            weights = view_scores.view(B, num_ref, 1, 1, 1)
            enhanced_query = (attended_features * weights).sum(dim=1)

        # Output projection
        enhanced_query = self.output_proj(enhanced_query)

        # Residual connection
        enhanced_query = enhanced_query + query_features

        return enhanced_query, view_scores, attention_maps

    def _cross_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head cross-attention between query and reference.

        Args:
            Q, K, V: (B, C, H, W)

        Returns:
            attended: (B, C, H, W)
            attention_map: (B, H, W)
        """
        B, C, H, W = Q.shape

        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, nh, HW, hd)
        K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, HW, HW)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, nh, HW, HW)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, nh, HW, hd)

        # Reshape back
        attended = attended.transpose(2, 3).contiguous().view(B, C, H, W)

        # Compute spatial attention map (average over heads and queries)
        attention_map = attention_weights.mean(dim=1).mean(dim=1).view(B, H, W)  # (B, H, W)

        return attended, attention_map
