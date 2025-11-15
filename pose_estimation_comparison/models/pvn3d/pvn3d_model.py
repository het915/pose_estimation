"""PVN3D model for per-object 6D pose estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .pointnet2 import PointNet2Backbone


class PVN3DModel(nn.Module):
    """
    PVN3D: Point-wise Voting Network for 6D Pose Estimation.

    Requires per-object training with keypoint supervision.
    Strong performance on occluded objects but no novel object generalization.
    """

    def __init__(
        self,
        num_points: int = 1024,
        num_keypoints: int = 8,
        pointnet_features: list = [64, 128, 256, 512],
        mlp_features: list = [512, 256, 128],
        use_rgb: bool = True,
    ):
        """
        Args:
            num_points: Number of input points
            num_keypoints: Number of keypoints to predict
            pointnet_features: Feature dimensions for PointNet++ layers
            mlp_features: MLP feature dimensions
            use_rgb: Whether to use RGB features
        """
        super(PVN3DModel, self).__init__()

        self.num_points = num_points
        self.num_keypoints = num_keypoints
        self.use_rgb = use_rgb

        # Input channels: xyz (3) + rgb (3) if used
        input_channels = 3 if not use_rgb else 6

        # PointNet++ backbone
        self.backbone = PointNet2Backbone(
            input_channels=input_channels,
            feature_dims=pointnet_features,
        )

        # Feature dimension from backbone
        backbone_feat_dim = pointnet_features[-1]

        # Segmentation head (object vs background)
        self.seg_head = nn.Sequential(
            nn.Conv1d(backbone_feat_dim, mlp_features[0], 1),
            nn.BatchNorm1d(mlp_features[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[0], mlp_features[1], 1),
            nn.BatchNorm1d(mlp_features[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[1], 2, 1),  # Binary segmentation
        )

        # Center prediction head
        self.center_head = nn.Sequential(
            nn.Conv1d(backbone_feat_dim, mlp_features[0], 1),
            nn.BatchNorm1d(mlp_features[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[0], mlp_features[1], 1),
            nn.BatchNorm1d(mlp_features[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[1], 3, 1),  # 3D offset to center
        )

        # Keypoint voting head
        self.keypoint_head = nn.Sequential(
            nn.Conv1d(backbone_feat_dim, mlp_features[0], 1),
            nn.BatchNorm1d(mlp_features[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[0], mlp_features[1], 1),
            nn.BatchNorm1d(mlp_features[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(mlp_features[1], num_keypoints * 3, 1),  # 3D offset to each keypoint
        )

    def forward(self, point_cloud: torch.Tensor, rgb: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            point_cloud: (B, N, 3) point cloud
            rgb: (B, N, 3) RGB colors (optional)

        Returns:
            Dictionary with predictions:
                - segmentation: (B, 2, N) object segmentation logits
                - center_offset: (B, 3, N) offset to object center
                - keypoint_offset: (B, num_keypoints*3, N) offset to keypoints
                - features: (B, C, N) point features
        """
        B, N, _ = point_cloud.shape

        # Combine point cloud with RGB if available
        if self.use_rgb and rgb is not None:
            # point_features: (B, N, 6)
            point_features = torch.cat([point_cloud, rgb], dim=-1)
        else:
            point_features = point_cloud

        # Transpose to (B, C, N) for PointNet++
        point_features = point_features.transpose(1, 2).contiguous()

        # Extract features with PointNet++
        features = self.backbone(point_features)  # (B, C, N)

        # Segmentation prediction
        seg_logits = self.seg_head(features)  # (B, 2, N)

        # Center offset prediction
        center_offset = self.center_head(features)  # (B, 3, N)

        # Keypoint offset prediction
        keypoint_offset = self.keypoint_head(features)  # (B, num_kp*3, N)

        return {
            "segmentation": seg_logits,
            "center_offset": center_offset,
            "keypoint_offset": keypoint_offset,
            "features": features,
        }

    def predict_pose(
        self,
        point_cloud: torch.Tensor,
        rgb: torch.Tensor = None,
        model_keypoints: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict 6D pose from point cloud.

        Args:
            point_cloud: (B, N, 3)
            rgb: (B, N, 3) optional
            model_keypoints: (B, num_kp, 3) model keypoints

        Returns:
            rotation: (B, 3, 3)
            translation: (B, 3)
        """
        predictions = self.forward(point_cloud, rgb)

        # Get object segmentation
        seg_probs = F.softmax(predictions["segmentation"], dim=1)  # (B, 2, N)
        object_mask = seg_probs[:, 1] > 0.5  # (B, N)

        # Predict keypoints by voting
        keypoint_offset = predictions["keypoint_offset"]  # (B, num_kp*3, N)
        B, _, N = keypoint_offset.shape

        # Reshape to (B, num_kp, 3, N)
        keypoint_offset = keypoint_offset.view(B, self.num_keypoints, 3, N)

        # Compute voted keypoint positions
        point_cloud_expanded = point_cloud.unsqueeze(1)  # (B, 1, N, 3)
        keypoint_offset = keypoint_offset.permute(0, 1, 3, 2)  # (B, num_kp, N, 3)

        # Voted keypoint positions: point + offset
        voted_keypoints = point_cloud_expanded + keypoint_offset  # (B, num_kp, N, 3)

        # Average over object points
        object_mask_expanded = object_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

        # Masked average
        voted_keypoints_masked = voted_keypoints * object_mask_expanded.float()
        num_object_points = object_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        predicted_keypoints = voted_keypoints_masked.sum(dim=2) / (num_object_points + 1e-8)  # (B, num_kp, 3)

        # Estimate pose using PnP or least squares
        if model_keypoints is not None:
            rotation, translation = self._solve_pose(predicted_keypoints, model_keypoints)
        else:
            # Return identity pose
            rotation = torch.eye(3, device=point_cloud.device).unsqueeze(0).repeat(B, 1, 1)
            translation = torch.zeros(B, 3, device=point_cloud.device)

        return rotation, translation

    def _solve_pose(
        self,
        predicted_keypoints: torch.Tensor,
        model_keypoints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve pose using Kabsch algorithm (SVD-based alignment).

        Args:
            predicted_keypoints: (B, num_kp, 3) predicted keypoints in camera frame
            model_keypoints: (B, num_kp, 3) model keypoints in object frame

        Returns:
            rotation: (B, 3, 3)
            translation: (B, 3)
        """
        B = predicted_keypoints.shape[0]

        # Center the point sets
        pred_centroid = predicted_keypoints.mean(dim=1, keepdim=True)  # (B, 1, 3)
        model_centroid = model_keypoints.mean(dim=1, keepdim=True)  # (B, 1, 3)

        pred_centered = predicted_keypoints - pred_centroid  # (B, num_kp, 3)
        model_centered = model_keypoints - model_centroid  # (B, num_kp, 3)

        # Compute covariance matrix
        H = torch.bmm(model_centered.transpose(1, 2), pred_centered)  # (B, 3, 3)

        # SVD
        U, S, Vh = torch.linalg.svd(H)  # U: (B, 3, 3), Vh: (B, 3, 3)
        V = Vh.transpose(1, 2)

        # Rotation matrix
        rotation = torch.bmm(V, U.transpose(1, 2))  # (B, 3, 3)

        # Handle reflections
        det = torch.det(rotation)  # (B,)
        V_corrected = V.clone()
        V_corrected[det < 0, :, -1] *= -1
        rotation = torch.bmm(V_corrected, U.transpose(1, 2))

        # Translation
        translation = pred_centroid.squeeze(1) - torch.bmm(
            rotation, model_centroid.transpose(1, 2)
        ).squeeze(-1)  # (B, 3)

        return rotation, translation
