"""Pose refinement module for FoundationPose."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PoseRefinement(nn.Module):
    """Iterative pose refinement module."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_iterations: int = 5,
        pose_update_lr: float = 0.01,
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_iterations: Number of refinement iterations
            pose_update_lr: Learning rate for pose updates
        """
        super(PoseRefinement, self).__init__()

        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.pose_update_lr = pose_update_lr

        # Pose update network
        self.update_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6),  # 6D pose update (rotation + translation)
        )

    def forward(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
        initial_rotation: torch.Tensor,
        initial_translation: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine pose iteratively.

        Args:
            query_features: (B, C, H, W) query features
            reference_features: (B, num_ref, C, H, W) reference features
            initial_rotation: (B, 3, 3) initial rotation
            initial_translation: (B, 3) initial translation
            camera_intrinsics: (B, 3, 3) camera intrinsics

        Returns:
            refined_rotation: (B, 3, 3)
            refined_translation: (B, 3)
        """
        B = query_features.shape[0]

        # Current pose estimate
        current_rotation = initial_rotation
        current_translation = initial_translation

        # Global feature pooling
        query_global = F.adaptive_avg_pool2d(query_features, 1).view(B, -1)  # (B, C)

        # Average reference features
        ref_global = F.adaptive_avg_pool2d(
            reference_features.mean(dim=1), 1
        ).view(B, -1)  # (B, C)

        # Iterative refinement
        for i in range(self.num_iterations):
            # Concatenate features
            combined_features = torch.cat([query_global, ref_global], dim=1)  # (B, 2*C)

            # Predict pose update
            pose_update = self.update_net(combined_features)  # (B, 6)

            # Split into rotation and translation updates
            rotation_update = pose_update[:, :3]  # (B, 3) - axis-angle
            translation_update = pose_update[:, 3:]  # (B, 3)

            # Convert axis-angle to rotation matrix
            rotation_delta = self._axis_angle_to_rotation_matrix(rotation_update)  # (B, 3, 3)

            # Update pose
            current_rotation = torch.bmm(rotation_delta, current_rotation)
            current_translation = current_translation + self.pose_update_lr * translation_update

        return current_rotation, current_translation

    def _axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix using Rodrigues formula.

        Args:
            axis_angle: (B, 3) axis-angle vectors

        Returns:
            rotation_matrix: (B, 3, 3)
        """
        B = axis_angle.shape[0]
        device = axis_angle.device

        # Compute angle
        angle = torch.norm(axis_angle, dim=1, keepdim=True)  # (B, 1)
        angle = torch.clamp(angle, min=1e-7)

        # Normalize axis
        axis = axis_angle / angle  # (B, 3)

        # Skew-symmetric matrix
        zeros = torch.zeros(B, device=device)
        K = torch.stack([
            torch.stack([zeros, -axis[:, 2], axis[:, 1]], dim=1),
            torch.stack([axis[:, 2], zeros, -axis[:, 0]], dim=1),
            torch.stack([-axis[:, 1], axis[:, 0], zeros], dim=1),
        ], dim=1)  # (B, 3, 3)

        # Rodrigues formula: R = I + sin(θ)K + (1 - cos(θ))K^2
        I = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)  # (B, 3, 3)

        sin_angle = torch.sin(angle).unsqueeze(-1)  # (B, 1, 1)
        cos_angle = torch.cos(angle).unsqueeze(-1)  # (B, 1, 1)

        R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)

        return R
