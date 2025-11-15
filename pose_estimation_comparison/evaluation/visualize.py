"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Dict
import torch


def visualize_pose(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_intrinsics: np.ndarray,
    model_points: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize 6D pose on image.

    Args:
        image: (H, W, 3) RGB image
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
        camera_intrinsics: (3, 3) camera intrinsics
        model_points: (N, 3) optional object model points
        save_path: Optional path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Show image
    ax.imshow(image)

    # Draw coordinate frame
    axis_length = 0.1  # meters
    axes_3d = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],  # X - Red
        [0, axis_length, 0],  # Y - Green
        [0, 0, axis_length],  # Z - Blue
    ])

    # Transform to camera frame
    axes_camera = (rotation @ axes_3d.T).T + translation

    # Project to image
    axes_2d = project_points_3d_to_2d(axes_camera, camera_intrinsics)

    # Draw axes
    origin = axes_2d[0]
    ax.plot([origin[0], axes_2d[1][0]], [origin[1], axes_2d[1][1]], 'r-', linewidth=3, label='X')
    ax.plot([origin[0], axes_2d[2][0]], [origin[1], axes_2d[2][1]], 'g-', linewidth=3, label='Y')
    ax.plot([origin[0], axes_2d[3][0]], [origin[1], axes_2d[3][1]], 'b-', linewidth=3, label='Z')

    # Draw model points if provided
    if model_points is not None:
        points_camera = (rotation @ model_points.T).T + translation
        points_2d = project_points_3d_to_2d(points_camera, camera_intrinsics)

        ax.scatter(points_2d[:, 0], points_2d[:, 1], c='yellow', s=1, alpha=0.5)

    ax.legend()
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_maps(
    image: np.ndarray,
    attention_maps: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Visualize attention maps.

    Args:
        image: (H, W, 3) RGB image
        attention_maps: Dictionary of attention maps
            - "spatial_attention": (H, W)
            - "cross_modal_attention": (H, W)
            - "cross_reference_attention": (num_ref, H, W)
        save_path: Optional path to save visualization
    """
    num_plots = 1 + len([k for k in attention_maps if attention_maps[k] is not None])

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    idx = 0

    # Original image
    axes[idx].imshow(image)
    axes[idx].set_title("Input Image")
    axes[idx].axis('off')
    idx += 1

    # Spatial attention
    if "spatial_attention" in attention_maps and attention_maps["spatial_attention"] is not None:
        spatial_attn = attention_maps["spatial_attention"]
        if isinstance(spatial_attn, torch.Tensor):
            spatial_attn = spatial_attn.cpu().numpy()

        axes[idx].imshow(image)
        axes[idx].imshow(spatial_attn, alpha=0.6, cmap='jet')
        axes[idx].set_title("Spatial Attention")
        axes[idx].axis('off')
        idx += 1

    # Cross-modal attention
    if "cross_modal_attention" in attention_maps and attention_maps["cross_modal_attention"] is not None:
        cross_modal = attention_maps["cross_modal_attention"]
        if isinstance(cross_modal, torch.Tensor):
            cross_modal = cross_modal.cpu().numpy()

        axes[idx].imshow(image)
        axes[idx].imshow(cross_modal, alpha=0.6, cmap='jet')
        axes[idx].set_title("Cross-Modal Attention")
        axes[idx].axis('off')
        idx += 1

    # Cross-reference attention (show best view)
    if "cross_reference_attention" in attention_maps and attention_maps["cross_reference_attention"] is not None:
        cross_ref = attention_maps["cross_reference_attention"]
        if isinstance(cross_ref, torch.Tensor):
            cross_ref = cross_ref.cpu().numpy()

        # Average over reference views
        if len(cross_ref.shape) == 3:  # (num_ref, H, W)
            cross_ref = cross_ref.mean(axis=0)

        axes[idx].imshow(image)
        axes[idx].imshow(cross_ref, alpha=0.6, cmap='jet')
        axes[idx].set_title("Cross-Reference Attention")
        axes[idx].axis('off')
        idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def project_points_3d_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Project 3D points to 2D image plane.

    Args:
        points_3d: (N, 3) 3D points in camera frame
        camera_intrinsics: (3, 3) camera intrinsics

    Returns:
        points_2d: (N, 2) 2D projected points
    """
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2] + 1e-8

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    points_2d = np.stack([u, v], axis=1)

    return points_2d


def plot_occlusion_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Plot comparison across occlusion levels.

    Args:
        results: Dictionary mapping method names to results
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics_to_plot = [
        ("add_accuracy", "ADD Accuracy (%)"),
        ("adds_accuracy", "ADD-S Accuracy (%)"),
        ("rotation_error_mean", "Rotation Error (degrees)"),
        ("translation_error_mean", "Translation Error (m)"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        for method_name, method_results in results.items():
            occlusion_levels = []
            metric_values = []

            for occ_range, metrics in sorted(method_results["by_occlusion"].items()):
                occlusion_levels.append(occ_range)
                metric_values.append(metrics[metric_key])

            ax.plot(occlusion_levels, metric_values, marker='o', label=method_name, linewidth=2)

        ax.set_xlabel("Occlusion Level")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
