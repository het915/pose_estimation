"""Spatial attention module for occlusion-aware feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on visible object regions.

    Uses depth information to identify occluded regions and
    reweights features accordingly.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SpatialAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Multi-head self-attention
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.key_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # Occlusion mask prediction from depth
        self.occlusion_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),  # Visibility score [0, 1]
        )

        self.output_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention.

        Args:
            rgb_features: (B, C, H, W) RGB features
            depth_features: (B, C, H, W) depth features

        Returns:
            attended_features: (B, C, H, W) attention-weighted features
            attention_map: (B, H, W) spatial attention map
        """
        B, C, H, W = rgb_features.shape

        # Predict visibility/occlusion from depth
        visibility_map = self.occlusion_predictor(depth_features)  # (B, 1, H, W)

        # Multi-head self-attention on RGB features
        Q = self.query_proj(rgb_features)  # (B, C, H, W)
        K = self.key_proj(rgb_features)
        V = self.value_proj(rgb_features)

        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, num_heads, H*W, head_dim)
        K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, H*W, H*W)

        # Apply visibility mask to attention scores
        visibility_flat = visibility_map.view(B, 1, 1, H * W)  # (B, 1, 1, H*W)
        attention_scores = attention_scores * visibility_flat  # Down-weight occluded regions

        # Softmax attention
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, H*W, H*W)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, num_heads, H*W, head_dim)

        # Reshape back
        attended = attended.transpose(2, 3).contiguous().view(B, C, H, W)  # (B, C, H, W)

        # Output projection
        attended = self.output_proj(attended)
        attended = self.dropout(attended)

        # Residual connection
        output = rgb_features + attended

        # Aggregate attention map (average over heads)
        spatial_attention_map = attention_weights.mean(dim=1)  # (B, H*W, H*W)
        # Get self-attention (diagonal-like aggregation)
        spatial_attention_map = spatial_attention_map.mean(dim=1).view(B, H, W)  # (B, H, W)

        # Combine with visibility
        final_attention_map = spatial_attention_map * visibility_map.squeeze(1)  # (B, H, W)

        return output, final_attention_map
