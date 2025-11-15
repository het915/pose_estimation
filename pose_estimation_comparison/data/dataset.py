"""Dataset classes for pose estimation."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple


class PoseEstimationDataset(Dataset):
    """Dataset for 6D pose estimation with occlusion support."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        objects: Optional[List[str]] = None,
        occlusion_levels: Optional[List[float]] = None,
        transform=None,
        num_points: int = 1024,
        use_rgb: bool = True,
        use_depth: bool = True,
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train', 'val', or 'test'
            objects: List of object IDs to include
            occlusion_levels: List of occlusion levels to include [0.0-1.0]
            transform: Data augmentation transform
            num_points: Number of points to sample from point cloud
            use_rgb: Whether to load RGB images
            use_depth: Whether to load depth images
        """
        self.data_root = data_root
        self.split = split
        self.objects = objects or []
        self.occlusion_levels = occlusion_levels or [0.0, 0.2, 0.4, 0.6]
        self.transform = transform
        self.num_points = num_points
        self.use_rgb = use_rgb
        self.use_depth = use_depth

        # Load dataset metadata
        self.samples = self._load_samples()

        # Load object models
        self.object_models = self._load_object_models()

    def _load_samples(self) -> List[Dict]:
        """Load dataset samples from metadata."""
        metadata_path = os.path.join(self.data_root, self.split, "metadata.json")

        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found at {metadata_path}")
            return []

        with open(metadata_path, "r") as f:
            all_samples = json.load(f)

        # Filter samples by object and occlusion level
        filtered_samples = []
        for sample in all_samples:
            if self.objects and sample["object_id"] not in self.objects:
                continue

            occlusion = sample.get("occlusion_ratio", 0.0)
            if not any(abs(occlusion - level) < 0.1 for level in self.occlusion_levels):
                continue

            filtered_samples.append(sample)

        return filtered_samples

    def _load_object_models(self) -> Dict[str, Dict]:
        """Load 3D object models."""
        models = {}
        models_dir = os.path.join(self.data_root, "models")

        if not os.path.exists(models_dir):
            return models

        for obj_id in self.objects:
            model_path = os.path.join(models_dir, f"{obj_id}.npy")
            if os.path.exists(model_path):
                models[obj_id] = {
                    "points": np.load(model_path),
                    "diameter": self._compute_diameter(np.load(model_path)),
                }

        return models

    def _compute_diameter(self, points: np.ndarray) -> float:
        """Compute object diameter."""
        return np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load images
        data = {
            "object_id": sample["object_id"],
            "scene_id": sample.get("scene_id", 0),
            "occlusion_ratio": sample.get("occlusion_ratio", 0.0),
        }

        # RGB image
        if self.use_rgb:
            rgb_path = os.path.join(self.data_root, self.split, sample["rgb_path"])
            if os.path.exists(rgb_path):
                rgb = Image.open(rgb_path).convert("RGB")
                data["rgb"] = np.array(rgb)
            else:
                data["rgb"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # Depth image
        if self.use_depth:
            depth_path = os.path.join(self.data_root, self.split, sample["depth_path"])
            if os.path.exists(depth_path):
                depth = np.load(depth_path) if depth_path.endswith(".npy") else cv2.imread(depth_path, -1)
                data["depth"] = depth
            else:
                data["depth"] = np.zeros((480, 640), dtype=np.float32)

        # Camera intrinsics
        data["camera_intrinsics"] = np.array(sample["camera_intrinsics"])

        # Ground truth pose
        data["rotation"] = np.array(sample["rotation"])  # 3x3 rotation matrix
        data["translation"] = np.array(sample["translation"])  # 3x1 translation vector

        # Object model points
        if sample["object_id"] in self.object_models:
            data["model_points"] = self.object_models[sample["object_id"]]["points"]
            data["object_diameter"] = self.object_models[sample["object_id"]]["diameter"]
        else:
            data["model_points"] = np.zeros((100, 3), dtype=np.float32)
            data["object_diameter"] = 1.0

        # Generate point cloud from depth
        if self.use_depth and "depth" in data:
            data["point_cloud"] = self._depth_to_point_cloud(
                data["depth"], data["camera_intrinsics"]
            )

            # Sample points
            if data["point_cloud"].shape[0] > self.num_points:
                indices = np.random.choice(
                    data["point_cloud"].shape[0], self.num_points, replace=False
                )
                data["point_cloud"] = data["point_cloud"][indices]
            elif data["point_cloud"].shape[0] < self.num_points:
                # Pad with zeros
                pad_size = self.num_points - data["point_cloud"].shape[0]
                data["point_cloud"] = np.vstack([
                    data["point_cloud"],
                    np.zeros((pad_size, 3))
                ])

        # Apply augmentation
        if self.transform:
            data = self.transform(data)

        # Convert to tensors
        return self._to_tensor(data)

    def _depth_to_point_cloud(
        self, depth: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        """Convert depth image to point cloud."""
        h, w = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create meshgrid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack and filter valid points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < 10.0)  # Filter outliers

        return points[valid]

    def _to_tensor(self, data: Dict) -> Dict:
        """Convert numpy arrays to tensors."""
        tensor_data = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if key == "rgb":
                    # Convert HWC to CHW and normalize
                    value = value.transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor_data[key] = torch.from_numpy(value.astype(np.float32))
            else:
                tensor_data[key] = value

        return tensor_data


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    if len(batch) == 0:
        return {}

    collated = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]

    return collated
