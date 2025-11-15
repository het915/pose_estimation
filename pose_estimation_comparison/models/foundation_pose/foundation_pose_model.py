"""FoundationPose model for novel object 6D pose estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .encoder import ImageEncoder
from .refinement import PoseRefinement
from .renderer import DifferentiableRenderer


class FoundationPoseModel(nn.Module):
    """
    FoundationPose: Novel object 6D pose estimation using multi-view references.

    Key features:
    - Zero-shot generalization to novel objects
    - Multi-view reference matching
    - Iterative pose refinement
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 256,
        num_reference_views: int = 8,
        num_refinement_iterations: int = 5,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            backbone: Backbone architecture
            feature_dim: Feature dimension
            num_reference_views: Number of reference views
            num_refinement_iterations: Number of refinement iterations
            image_size: Input image size
        """
        super(FoundationPoseModel, self).__init__()

        self.feature_dim = feature_dim
        self.num_reference_views = num_reference_views
        self.num_refinement_iterations = num_refinement_iterations
        self.image_size = image_size

        # Image encoder
        self.encoder = ImageEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=True,
        )

        # Pose refinement module
        self.refinement = PoseRefinement(
            feature_dim=feature_dim,
            num_iterations=num_refinement_iterations,
        )

        # Differentiable renderer (simplified)
        self.renderer = DifferentiableRenderer(
            image_size=image_size,
        )

    def forward(
        self,
        query_rgb: torch.Tensor,
        query_depth: torch.Tensor,
        reference_rgbs: torch.Tensor,
        reference_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        initial_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_rgb: (B, 3, H, W) query RGB image
            query_depth: (B, 1, H, W) query depth image
            reference_rgbs: (B, num_ref, 3, H, W) reference RGB images
            reference_poses: (B, num_ref, 4, 4) reference poses
            camera_intrinsics: (B, 3, 3) camera intrinsics
            initial_pose: Optional initial pose estimate (R, t)

        Returns:
            Dictionary with predictions
        """
        B = query_rgb.shape[0]

        # Encode query image
        query_features = self.encoder(torch.cat([query_rgb, query_depth], dim=1))  # (B, C, H', W')

        # Encode reference images
        ref_features_list = []
        for i in range(self.num_reference_views):
            ref_rgb = reference_rgbs[:, i]  # (B, 3, H, W)
            ref_feat = self.encoder(ref_rgb)  # (B, C, H', W')
            ref_features_list.append(ref_feat)

        reference_features = torch.stack(ref_features_list, dim=1)  # (B, num_ref, C, H', W')

        # Match query to references
        matching_scores = self._compute_matching_scores(query_features, reference_features)  # (B, num_ref)

        # Initial pose estimate
        if initial_pose is None:
            # Use weighted average of reference poses
            rotation, translation = self._aggregate_reference_poses(
                reference_poses, matching_scores
            )
        else:
            rotation, translation = initial_pose

        # Iterative refinement
        refined_rotation, refined_translation = self.refinement(
            query_features=query_features,
            reference_features=reference_features,
            initial_rotation=rotation,
            initial_translation=translation,
            camera_intrinsics=camera_intrinsics,
        )

        return {
            "rotation": refined_rotation,  # (B, 3, 3)
            "translation": refined_translation,  # (B, 3)
            "matching_scores": matching_scores,  # (B, num_ref)
            "query_features": query_features,
            "reference_features": reference_features,
        }

    def _compute_matching_scores(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching scores between query and reference features.

        Args:
            query_features: (B, C, H, W)
            reference_features: (B, num_ref, C, H, W)

        Returns:
            matching_scores: (B, num_ref)
        """
        B, C, H, W = query_features.shape
        num_ref = reference_features.shape[1]

        # Global average pooling
        query_global = F.adaptive_avg_pool2d(query_features, 1).view(B, C)  # (B, C)
        reference_global = F.adaptive_avg_pool2d(
            reference_features.view(B * num_ref, C, H, W), 1
        ).view(B, num_ref, C)  # (B, num_ref, C)

        # Cosine similarity
        query_norm = F.normalize(query_global, dim=1)  # (B, C)
        reference_norm = F.normalize(reference_global, dim=2)  # (B, num_ref, C)

        # Compute similarity
        matching_scores = torch.bmm(
            reference_norm,  # (B, num_ref, C)
            query_norm.unsqueeze(-1),  # (B, C, 1)
        ).squeeze(-1)  # (B, num_ref)

        # Softmax to get weights
        matching_scores = F.softmax(matching_scores, dim=1)

        return matching_scores

    def _aggregate_reference_poses(
        self,
        reference_poses: torch.Tensor,
        matching_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate reference poses using matching scores.

        Args:
            reference_poses: (B, num_ref, 4, 4) reference poses
            matching_scores: (B, num_ref) matching scores

        Returns:
            rotation: (B, 3, 3)
            translation: (B, 3)
        """
        B, num_ref = matching_scores.shape

        # Extract rotations and translations
        reference_rotations = reference_poses[:, :, :3, :3]  # (B, num_ref, 3, 3)
        reference_translations = reference_poses[:, :, :3, 3]  # (B, num_ref, 3)

        # Weighted average of translations
        weights = matching_scores.unsqueeze(-1)  # (B, num_ref, 1)
        translation = (reference_translations * weights).sum(dim=1)  # (B, 3)

        # For rotation, use the highest scoring reference (simpler than quaternion averaging)
        best_ref_idx = matching_scores.argmax(dim=1)  # (B,)
        rotation = reference_rotations[torch.arange(B), best_ref_idx]  # (B, 3, 3)

        return rotation, translation

    def predict_pose(
        self,
        query_rgb: torch.Tensor,
        query_depth: torch.Tensor,
        reference_rgbs: torch.Tensor,
        reference_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict 6D pose.

        Returns:
            rotation: (B, 3, 3)
            translation: (B, 3)
        """
        output = self.forward(
            query_rgb=query_rgb,
            query_depth=query_depth,
            reference_rgbs=reference_rgbs,
            reference_poses=reference_poses,
            camera_intrinsics=camera_intrinsics,
        )

        return output["rotation"], output["translation"]
