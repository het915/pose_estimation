"""Pose estimation metrics."""

import numpy as np
import torch
from typing import Dict, List, Tuple


class PoseMetrics:
    """Compute pose estimation metrics."""

    def __init__(
        self,
        add_threshold: float = 0.1,
        adds_threshold: float = 0.1,
        rotation_threshold: float = 5.0,
        translation_threshold: float = 0.05,
    ):
        """
        Args:
            add_threshold: ADD threshold (fraction of object diameter)
            adds_threshold: ADD-S threshold (fraction of object diameter)
            rotation_threshold: Rotation error threshold (degrees)
            translation_threshold: Translation error threshold (meters)
        """
        self.add_threshold = add_threshold
        self.adds_threshold = adds_threshold
        self.rotation_threshold = rotation_threshold
        self.translation_threshold = translation_threshold

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.rotation_errors = []
        self.translation_errors = []
        self.add_errors = []
        self.adds_errors = []
        self.object_diameters = []
        self.occlusion_ratios = []

    def update(
        self,
        pred_rotation: np.ndarray,
        pred_translation: np.ndarray,
        gt_rotation: np.ndarray,
        gt_translation: np.ndarray,
        model_points: np.ndarray,
        object_diameter: float,
        occlusion_ratio: float = 0.0,
    ):
        """
        Update metrics with a new prediction.

        Args:
            pred_rotation: (3, 3) predicted rotation
            pred_translation: (3,) predicted translation
            gt_rotation: (3, 3) ground truth rotation
            gt_translation: (3,) ground truth translation
            model_points: (N, 3) object model points
            object_diameter: Object diameter
            occlusion_ratio: Occlusion ratio [0, 1]
        """
        # Rotation error
        rot_error = compute_rotation_error(pred_rotation, gt_rotation)
        self.rotation_errors.append(rot_error)

        # Translation error
        trans_error = compute_translation_error(pred_translation, gt_translation)
        self.translation_errors.append(trans_error)

        # ADD error
        add_error = compute_add(
            pred_rotation, pred_translation,
            gt_rotation, gt_translation,
            model_points
        )
        self.add_errors.append(add_error)

        # ADD-S error
        adds_error = compute_adds(
            pred_rotation, pred_translation,
            gt_rotation, gt_translation,
            model_points
        )
        self.adds_errors.append(adds_error)

        # Metadata
        self.object_diameters.append(object_diameter)
        self.occlusion_ratios.append(occlusion_ratio)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric values
        """
        rotation_errors = np.array(self.rotation_errors)
        translation_errors = np.array(self.translation_errors)
        add_errors = np.array(self.add_errors)
        adds_errors = np.array(self.adds_errors)
        object_diameters = np.array(self.object_diameters)

        metrics = {}

        # Mean errors
        metrics["rotation_error_mean"] = np.mean(rotation_errors)
        metrics["rotation_error_median"] = np.median(rotation_errors)
        metrics["translation_error_mean"] = np.mean(translation_errors)
        metrics["translation_error_median"] = np.median(translation_errors)

        # ADD accuracy
        add_thresholds = add_errors / object_diameters
        metrics["add_accuracy"] = np.mean(add_thresholds < self.add_threshold) * 100

        # ADD-S accuracy
        adds_thresholds = adds_errors / object_diameters
        metrics["adds_accuracy"] = np.mean(adds_thresholds < self.adds_threshold) * 100

        # Rotation accuracy
        metrics["rotation_accuracy"] = np.mean(rotation_errors < self.rotation_threshold) * 100

        # Translation accuracy
        metrics["translation_accuracy"] = np.mean(translation_errors < self.translation_threshold) * 100

        return metrics

    def compute_by_occlusion(self, occlusion_bins: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics binned by occlusion level.

        Args:
            occlusion_bins: List of occlusion bin edges [0.0, 0.2, 0.4, 0.6, 0.8]

        Returns:
            Dictionary mapping occlusion ranges to metrics
        """
        occlusion_ratios = np.array(self.occlusion_ratios)
        rotation_errors = np.array(self.rotation_errors)
        translation_errors = np.array(self.translation_errors)
        add_errors = np.array(self.add_errors)
        adds_errors = np.array(self.adds_errors)
        object_diameters = np.array(self.object_diameters)

        results = {}

        for i in range(len(occlusion_bins) - 1):
            bin_min = occlusion_bins[i]
            bin_max = occlusion_bins[i + 1]

            # Filter samples in this bin
            mask = (occlusion_ratios >= bin_min) & (occlusion_ratios < bin_max)

            if np.sum(mask) == 0:
                continue

            bin_name = f"{int(bin_min*100)}-{int(bin_max*100)}%"

            # Compute metrics for this bin
            add_thresholds = add_errors[mask] / object_diameters[mask]
            adds_thresholds = adds_errors[mask] / object_diameters[mask]

            results[bin_name] = {
                "num_samples": int(np.sum(mask)),
                "rotation_error_mean": float(np.mean(rotation_errors[mask])),
                "translation_error_mean": float(np.mean(translation_errors[mask])),
                "add_accuracy": float(np.mean(add_thresholds < self.add_threshold) * 100),
                "adds_accuracy": float(np.mean(adds_thresholds < self.adds_threshold) * 100),
            }

        return results


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees.

    Args:
        R_pred: (3, 3) predicted rotation
        R_gt: (3, 3) ground truth rotation

    Returns:
        Rotation error in degrees
    """
    # Compute relative rotation
    R_error = R_pred @ R_gt.T

    # Compute angle from rotation matrix
    trace = np.trace(R_error)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    # Convert to degrees
    return np.degrees(angle)


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute translation error (Euclidean distance).

    Args:
        t_pred: (3,) predicted translation
        t_gt: (3,) ground truth translation

    Returns:
        Translation error in meters
    """
    return np.linalg.norm(t_pred - t_gt)


def compute_add(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """
    Compute Average Distance of Model Points (ADD) metric.

    Args:
        R_pred, t_pred: Predicted pose
        R_gt, t_gt: Ground truth pose
        model_points: (N, 3) object model points

    Returns:
        ADD error (mean distance)
    """
    # Transform model points
    points_pred = (R_pred @ model_points.T).T + t_pred
    points_gt = (R_gt @ model_points.T).T + t_gt

    # Compute distances
    distances = np.linalg.norm(points_pred - points_gt, axis=1)

    return np.mean(distances)


def compute_adds(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """
    Compute Average Distance of Model Points Symmetric (ADD-S) metric.

    For symmetric objects, finds closest point matches.

    Args:
        R_pred, t_pred: Predicted pose
        R_gt, t_gt: Ground truth pose
        model_points: (N, 3) object model points

    Returns:
        ADD-S error (mean minimum distance)
    """
    # Transform model points
    points_pred = (R_pred @ model_points.T).T + t_pred
    points_gt = (R_gt @ model_points.T).T + t_gt

    # Compute pairwise distances
    distances = np.linalg.norm(
        points_pred[:, None, :] - points_gt[None, :, :],
        axis=2
    )

    # Find minimum distance for each predicted point
    min_distances = np.min(distances, axis=1)

    return np.mean(min_distances)
