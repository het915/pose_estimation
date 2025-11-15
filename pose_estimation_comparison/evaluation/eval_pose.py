"""Pose evaluation script."""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.data import DataLoader

from .metrics import PoseMetrics


class PoseEvaluator:
    """Evaluator for pose estimation models."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        occlusion_bins: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ):
        """
        Args:
            model: Pose estimation model
            dataloader: Data loader
            device: Device to run evaluation on
            occlusion_bins: Occlusion bins for analysis
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.occlusion_bins = occlusion_bins

        self.model.to(device)
        self.model.eval()

        self.metrics = PoseMetrics()

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dictionary of metrics
        """
        self.metrics.reset()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Move to device
                batch = self._to_device(batch)

                # Get predictions
                predictions = self._predict(batch)

                # Update metrics
                self._update_metrics(predictions, batch)

        # Compute overall metrics
        overall_metrics = self.metrics.compute()

        # Compute occlusion-specific metrics
        occlusion_metrics = self.metrics.compute_by_occlusion(self.occlusion_bins)

        return {
            "overall": overall_metrics,
            "by_occlusion": occlusion_metrics,
        }

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _predict(self, batch: Dict) -> Dict:
        """Get model predictions."""
        # This is model-specific and should be overridden
        # Default implementation for models with predict_pose method
        if hasattr(self.model, "predict_pose"):
            if "reference_rgbs" in batch:
                # FoundationPose or AttentionPose
                rotation, translation = self.model.predict_pose(
                    query_rgb=batch["rgb"],
                    query_depth=batch["depth"],
                    reference_rgbs=batch["reference_rgbs"],
                    reference_poses=batch["reference_poses"],
                    camera_intrinsics=batch["camera_intrinsics"],
                )[:2]  # Ignore confidence if present
            else:
                # PVN3D
                rotation, translation = self.model.predict_pose(
                    point_cloud=batch["point_cloud"],
                    rgb=batch.get("rgb_points"),
                    model_keypoints=batch.get("model_keypoints"),
                )

            return {
                "rotation": rotation,
                "translation": translation,
            }
        else:
            raise NotImplementedError("Model must have predict_pose method")

    def _update_metrics(self, predictions: Dict, batch: Dict):
        """Update metrics with predictions."""
        B = predictions["rotation"].shape[0]

        # Convert to numpy
        pred_rotation = predictions["rotation"].cpu().numpy()
        pred_translation = predictions["translation"].cpu().numpy()
        gt_rotation = batch["rotation"].cpu().numpy()
        gt_translation = batch["translation"].cpu().numpy()

        for i in range(B):
            self.metrics.update(
                pred_rotation=pred_rotation[i],
                pred_translation=pred_translation[i],
                gt_rotation=gt_rotation[i],
                gt_translation=gt_translation[i],
                model_points=batch["model_points"][i].cpu().numpy(),
                object_diameter=batch["object_diameter"][i].item(),
                occlusion_ratio=batch["occlusion_ratio"][i].item() if "occlusion_ratio" in batch else 0.0,
            )
