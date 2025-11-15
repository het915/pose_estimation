"""Cross-modal attention for RGB-Depth fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing RGB and Depth features.

    Allows RGB and Depth to attend to each other,
    creating a richer fused representation.
    """

    def __init__(
        self,
        rgb_dim: int = 256,
        depth_dim: int = 256,
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            rgb_dim: RGB feature dimension
            depth_dim: Depth feature dimension
            fusion_dim: Fused feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossModalAttention, self).__init__()

        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads

        assert fusion_dim % num_heads == 0

        # RGB attends to Depth
        self.rgb_to_depth_query = nn.Conv2d(rgb_dim, fusion_dim, 1)
        self.depth_key = nn.Conv2d(depth_dim, fusion_dim, 1)
        self.depth_value = nn.Conv2d(depth_dim, fusion_dim, 1)

        # Depth attends to RGB
        self.depth_to_rgb_query = nn.Conv2d(depth_dim, fusion_dim, 1)
        self.rgb_key = nn.Conv2d(rgb_dim, fusion_dim, 1)
        self.rgb_value = nn.Conv2d(rgb_dim, fusion_dim, 1)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(fusion_dim, fusion_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            rgb_features: (B, C_rgb, H, W)
            depth_features: (B, C_depth, H, W)

        Returns:
            fused_features: (B, fusion_dim, H, W)
            attention_map: (B, H, W) cross-modal attention
        """
        B, _, H, W = rgb_features.shape

        # RGB attends to Depth
        Q_rgb = self.rgb_to_depth_query(rgb_features)  # (B, fusion_dim, H, W)
        K_depth = self.depth_key(depth_features)
        V_depth = self.depth_value(depth_features)

        rgb_attended = self._multi_head_attention(Q_rgb, K_depth, V_depth)  # (B, fusion_dim, H, W)

        # Depth attends to RGB
        Q_depth = self.depth_to_rgb_query(depth_features)
        K_rgb = self.rgb_key(rgb_features)
        V_rgb = self.rgb_value(rgb_features)

        depth_attended = self._multi_head_attention(Q_depth, K_rgb, V_rgb)

        # Fuse both attended features
        fused = torch.cat([rgb_attended, depth_attended], dim=1)  # (B, 2*fusion_dim, H, W)
        fused_features = self.fusion(fused)  # (B, fusion_dim, H, W)

        # Compute attention map for visualization (simplified)
        attention_map = self._compute_attention_map(Q_rgb, K_depth, H, W)

        return fused_features, attention_map

    def _multi_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-head attention.

        Args:
            Q, K, V: (B, C, H, W)

        Returns:
            attended: (B, C, H, W)
        """
        B, C, H, W = Q.shape

        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, nh, HW, hd)
        K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, HW, HW)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)  # (B, nh, HW, hd)

        # Reshape back
        attended = attended.transpose(2, 3).contiguous().view(B, C, H, W)

        return attended

    def _compute_attention_map(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute attention map for visualization."""
        B, C = Q.shape[:2]

        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)

        # Simplified: global attention
        Q_global = Q.mean(dim=-1)  # (B, nh, hd)
        K_global = K.mean(dim=-1)  # (B, nh, hd)

        attention = torch.sum(Q_global * K_global, dim=-1)  # (B, nh)
        attention = attention.mean(dim=1)  # (B,)

        # Create spatial map (uniform for simplicity)
        attention_map = attention.view(B, 1, 1).expand(B, H, W)

        return attention_map
