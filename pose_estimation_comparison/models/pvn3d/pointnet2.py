"""PointNet++ backbone for PVN3D."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet2Backbone(nn.Module):
    """Simplified PointNet++ backbone for feature extraction."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_dims: list = [64, 128, 256, 512],
    ):
        """
        Args:
            input_channels: Number of input channels (3 for xyz, 6 for xyz+rgb)
            feature_dims: Feature dimensions for each layer
        """
        super(PointNet2Backbone, self).__init__()

        self.input_channels = input_channels
        self.feature_dims = feature_dims

        # Build layers
        self.layers = nn.ModuleList()

        in_channels = input_channels
        for out_channels in feature_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

        # Global feature extraction
        self.global_feat = nn.Sequential(
            nn.Conv1d(feature_dims[-1], feature_dims[-1], 1),
            nn.BatchNorm1d(feature_dims[-1]),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, N) input features

        Returns:
            features: (B, C_out, N) output features
        """
        # Local features
        for layer in self.layers:
            x = layer(x)

        local_features = x  # (B, C, N)

        # Global features
        global_feat = self.global_feat(x)  # (B, C, N)
        global_feat = torch.max(global_feat, dim=2, keepdim=True)[0]  # (B, C, 1)
        global_feat = global_feat.expand_as(local_features)  # (B, C, N)

        # Concatenate local and global features
        features = local_features + global_feat  # Skip connection

        return features
