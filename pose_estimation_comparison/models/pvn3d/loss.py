"""Loss functions for PVN3D."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PVN3DLoss(nn.Module):
    """Combined loss for PVN3D training."""

    def __init__(
        self,
        keypoint_weight: float = 1.0,
        segmentation_weight: float = 0.5,
        center_weight: float = 1.0,
    ):
        """
        Args:
            keypoint_weight: Weight for keypoint loss
            segmentation_weight: Weight for segmentation loss
            center_weight: Weight for center loss
        """
        super(PVN3DLoss, self).__init__()

        self.keypoint_weight = keypoint_weight
        self.segmentation_weight = segmentation_weight
        self.center_weight = center_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Segmentation loss
        if "segmentation" in predictions and "segmentation" in targets:
            seg_loss = F.cross_entropy(
                predictions["segmentation"],
                targets["segmentation"],
            )
            losses["segmentation_loss"] = seg_loss

        # Center offset loss
        if "center_offset" in predictions and "center_offset" in targets:
            # Only compute loss on object points
            object_mask = targets.get("object_mask", None)

            if object_mask is not None:
                center_pred = predictions["center_offset"]  # (B, 3, N)
                center_target = targets["center_offset"]  # (B, 3, N)

                # Masked L1 loss
                center_loss = F.l1_loss(
                    center_pred * object_mask.unsqueeze(1),
                    center_target * object_mask.unsqueeze(1),
                    reduction='sum'
                ) / (object_mask.sum() + 1e-8)
            else:
                center_loss = F.l1_loss(
                    predictions["center_offset"],
                    targets["center_offset"],
                )

            losses["center_loss"] = center_loss

        # Keypoint offset loss
        if "keypoint_offset" in predictions and "keypoint_offset" in targets:
            object_mask = targets.get("object_mask", None)

            if object_mask is not None:
                kp_pred = predictions["keypoint_offset"]  # (B, num_kp*3, N)
                kp_target = targets["keypoint_offset"]  # (B, num_kp*3, N)

                # Masked L1 loss
                keypoint_loss = F.l1_loss(
                    kp_pred * object_mask.unsqueeze(1),
                    kp_target * object_mask.unsqueeze(1),
                    reduction='sum'
                ) / (object_mask.sum() + 1e-8)
            else:
                keypoint_loss = F.l1_loss(
                    predictions["keypoint_offset"],
                    targets["keypoint_offset"],
                )

            losses["keypoint_loss"] = keypoint_loss

        # Total loss
        total_loss = 0.0

        if "segmentation_loss" in losses:
            total_loss += self.segmentation_weight * losses["segmentation_loss"]

        if "center_loss" in losses:
            total_loss += self.center_weight * losses["center_loss"]

        if "keypoint_loss" in losses:
            total_loss += self.keypoint_weight * losses["keypoint_loss"]

        losses["total_loss"] = total_loss

        return losses
