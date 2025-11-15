"""Uncertainty estimation network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyNet(nn.Module):
    """
    Network for estimating pose uncertainty.

    Predicts confidence scores for pose estimates,
    which is particularly useful under heavy occlusion.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128, 64],
        output_dim: int = 7,  # 6D pose + confidence
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (default: 7 for 6D pose + confidence)
            dropout: Dropout rate
        """
        super(UncertaintyNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )

        # MLP for uncertainty estimation
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Separate heads for different uncertainty components
        self.rotation_uncertainty = nn.Linear(hidden_dims[-1], 3)
        self.translation_uncertainty = nn.Linear(hidden_dims[-1], 3)
        self.overall_confidence = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty from features.

        Args:
            features: (B, C, H, W) input features

        Returns:
            uncertainty: (B, output_dim) uncertainty estimates
                - First 3 values: rotation uncertainty (log variance)
                - Next 3 values: translation uncertainty (log variance)
                - Last value: overall confidence (logit)
        """
        B = features.shape[0]

        # Global pooling
        pooled = self.feature_extractor(features).view(B, -1)  # (B, C)

        # Extract intermediate features
        x = pooled
        for layer in self.mlp[:-1]:
            x = layer(x)

        # Predict uncertainty components
        rotation_uncertainty = self.rotation_uncertainty(x)  # (B, 3)
        translation_uncertainty = self.translation_uncertainty(x)  # (B, 3)
        confidence = self.overall_confidence(x)  # (B, 1)

        # Concatenate
        uncertainty = torch.cat([
            rotation_uncertainty,
            translation_uncertainty,
            confidence,
        ], dim=1)  # (B, 7)

        return uncertainty

    def compute_loss(
        self,
        uncertainty: torch.Tensor,
        rotation_error: torch.Tensor,
        translation_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncertainty loss.

        Args:
            uncertainty: (B, 7) predicted uncertainty
            rotation_error: (B, 3) rotation error
            translation_error: (B, 3) translation error

        Returns:
            loss: Scalar uncertainty loss
        """
        # Extract components
        rot_log_var = uncertainty[:, :3]  # (B, 3)
        trans_log_var = uncertainty[:, 3:6]  # (B, 3)

        # Rotation loss (negative log-likelihood)
        rot_loss = (rotation_error ** 2 / (2 * torch.exp(rot_log_var)) + 0.5 * rot_log_var).mean()

        # Translation loss (negative log-likelihood)
        trans_loss = (translation_error ** 2 / (2 * torch.exp(trans_log_var)) + 0.5 * trans_log_var).mean()

        # Total loss
        loss = rot_loss + trans_loss

        return loss
