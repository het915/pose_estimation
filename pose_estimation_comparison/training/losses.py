"""Combined loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CombinedLoss(nn.Module):
    """Combined loss for pose estimation."""

    def __init__(
        self,
        pose_weight: float = 1.0,
        feature_weight: float = 0.5,
        rendering_weight: float = 0.3,
        attention_weight: float = 0.2,
        uncertainty_weight: float = 0.1,
    ):
        """
        Args:
            pose_weight: Weight for pose loss
            feature_weight: Weight for feature matching loss
            rendering_weight: Weight for rendering loss
            attention_weight: Weight for attention regularization
            uncertainty_weight: Weight for uncertainty loss
        """
        super(CombinedLoss, self).__init__()

        self.pose_weight = pose_weight
        self.feature_weight = feature_weight
        self.rendering_weight = rendering_weight
        self.attention_weight = attention_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Pose loss (rotation + translation)
        if "rotation" in predictions and "rotation" in targets:
            rotation_loss = self._rotation_loss(
                predictions["rotation"], targets["rotation"]
            )
            translation_loss = F.mse_loss(
                predictions["translation"], targets["translation"]
            )

            losses["rotation_loss"] = rotation_loss
            losses["translation_loss"] = translation_loss
            losses["pose_loss"] = rotation_loss + translation_loss

        # Feature matching loss
        if "matching_scores" in predictions:
            # Encourage high matching scores (optional)
            matching_entropy = -(predictions["matching_scores"] *
                               torch.log(predictions["matching_scores"] + 1e-8)).sum(dim=1).mean()
            losses["matching_loss"] = matching_entropy

        # Attention regularization (encourage sparsity)
        if "spatial_attention" in predictions and predictions["spatial_attention"] is not None:
            spatial_attn = predictions["spatial_attention"]
            attention_reg = torch.mean(spatial_attn)  # Encourage sparsity
            losses["attention_reg"] = attention_reg

        # Uncertainty loss (if available)
        if "confidence" in predictions and "pose_error" in targets:
            # Higher confidence should correlate with lower error
            uncertainty_loss = F.mse_loss(
                predictions["confidence"],
                torch.exp(-targets["pose_error"])  # Convert error to confidence
            )
            losses["uncertainty_loss"] = uncertainty_loss

        # Total loss
        total_loss = 0.0

        if "pose_loss" in losses:
            total_loss += self.pose_weight * losses["pose_loss"]

        if "matching_loss" in losses:
            total_loss += self.feature_weight * losses["matching_loss"]

        if "attention_reg" in losses:
            total_loss += self.attention_weight * losses["attention_reg"]

        if "uncertainty_loss" in losses:
            total_loss += self.uncertainty_weight * losses["uncertainty_loss"]

        losses["total_loss"] = total_loss

        return losses

    def _rotation_loss(
        self,
        pred_rotation: torch.Tensor,
        gt_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rotation loss using geodesic distance.

        Args:
            pred_rotation: (B, 3, 3) predicted rotation
            gt_rotation: (B, 3, 3) ground truth rotation

        Returns:
            Rotation loss
        """
        # Compute relative rotation
        R_error = torch.bmm(pred_rotation, gt_rotation.transpose(1, 2))

        # Geodesic distance
        trace = R_error[:, 0, 0] + R_error[:, 1, 1] + R_error[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        return angle.mean()
