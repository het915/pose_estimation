"""Synthetic data generation utilities."""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class SyntheticDataGenerator:
    """Generate synthetic training data for pose estimation."""

    def __init__(
        self,
        output_dir: str,
        num_scenes: int = 1000,
        objects_per_scene: int = 1,
        occlusion_levels: List[float] = [0.0, 0.2, 0.4, 0.6],
    ):
        """
        Args:
            output_dir: Directory to save generated data
            num_scenes: Number of scenes to generate
            objects_per_scene: Number of objects per scene
            occlusion_levels: Target occlusion levels
        """
        self.output_dir = Path(output_dir)
        self.num_scenes = num_scenes
        self.objects_per_scene = objects_per_scene
        self.occlusion_levels = occlusion_levels

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(
        self,
        split: str = "train",
        object_models: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """
        Generate synthetic dataset.

        Args:
            split: Dataset split ('train', 'val', 'test')
            object_models: Dictionary of object models {obj_id: points}

        Returns:
            List of sample metadata
        """
        print(f"Generating {self.num_scenes} scenes for {split} split...")

        # Create split directory
        split_dir = self.output_dir / split
        split_dir.mkdir(exist_ok=True)

        (split_dir / "rgb").mkdir(exist_ok=True)
        (split_dir / "depth").mkdir(exist_ok=True)

        samples = []

        for scene_id in range(self.num_scenes):
            if scene_id % 100 == 0:
                print(f"Generated {scene_id}/{self.num_scenes} scenes")

            # Generate random camera parameters
            camera_intrinsics = self._generate_camera_intrinsics()

            # For each occlusion level
            for occlusion_level in self.occlusion_levels:
                # Generate scene
                sample = self._generate_scene(
                    scene_id=scene_id,
                    split=split,
                    camera_intrinsics=camera_intrinsics,
                    occlusion_level=occlusion_level,
                    object_models=object_models,
                )

                samples.append(sample)

        # Save metadata
        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"Generated {len(samples)} samples for {split} split")
        print(f"Metadata saved to {metadata_path}")

        return samples

    def _generate_camera_intrinsics(self) -> np.ndarray:
        """Generate random camera intrinsics."""
        # Typical RGB-D camera parameters
        fx = np.random.uniform(500, 600)
        fy = np.random.uniform(500, 600)
        cx = np.random.uniform(310, 330)
        cy = np.random.uniform(230, 250)

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return intrinsics

    def _generate_scene(
        self,
        scene_id: int,
        split: str,
        camera_intrinsics: np.ndarray,
        occlusion_level: float,
        object_models: Optional[Dict] = None,
    ) -> Dict:
        """Generate a single scene."""
        # Random object selection
        if object_models:
            object_id = np.random.choice(list(object_models.keys()))
        else:
            object_id = f"obj_{np.random.randint(1, 20):02d}"

        # Random pose
        rotation = self._generate_random_rotation()
        translation = self._generate_random_translation()

        # File paths
        rgb_filename = f"scene_{scene_id:06d}_occ_{int(occlusion_level*100):02d}_rgb.png"
        depth_filename = f"scene_{scene_id:06d}_occ_{int(occlusion_level*100):02d}_depth.npy"

        sample = {
            "scene_id": scene_id,
            "object_id": object_id,
            "rgb_path": f"rgb/{rgb_filename}",
            "depth_path": f"depth/{depth_filename}",
            "camera_intrinsics": camera_intrinsics.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "occlusion_ratio": occlusion_level,
        }

        return sample

    def _generate_random_rotation(self) -> np.ndarray:
        """Generate random rotation matrix."""
        from scipy.spatial.transform import Rotation as R

        # Random rotation
        angles = np.random.uniform(0, 360, size=3)
        rotation = R.from_euler('xyz', angles, degrees=True).as_matrix()

        return rotation

    def _generate_random_translation(self) -> np.ndarray:
        """Generate random translation vector."""
        # Random position within workspace
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.5, 1.5)  # Distance from camera

        translation = np.array([x, y, z])

        return translation

    def create_object_models(
        self,
        object_ids: List[str],
        num_points: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Create simple object models for testing.

        Args:
            object_ids: List of object IDs
            num_points: Number of points per model

        Returns:
            Dictionary of object models {obj_id: points}
        """
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        models = {}

        for obj_id in object_ids:
            # Generate random point cloud (simple box or sphere)
            if np.random.random() > 0.5:
                # Box
                points = self._generate_box_points(num_points)
            else:
                # Sphere
                points = self._generate_sphere_points(num_points)

            models[obj_id] = points

            # Save model
            model_path = models_dir / f"{obj_id}.npy"
            np.save(model_path, points)

        return models

    def _generate_box_points(self, num_points: int) -> np.ndarray:
        """Generate points on a box surface."""
        # Random box dimensions
        size = np.random.uniform(0.05, 0.15, size=3)

        points = []
        points_per_face = num_points // 6

        # Generate points on each face
        for i in range(3):
            for sign in [-1, 1]:
                face_points = np.random.uniform(-1, 1, size=(points_per_face, 3))
                face_points[:, i] = sign
                face_points *= size
                points.append(face_points)

        points = np.vstack(points)

        return points

    def _generate_sphere_points(self, num_points: int) -> np.ndarray:
        """Generate points on a sphere surface."""
        # Random sphere radius
        radius = np.random.uniform(0.05, 0.15)

        # Fibonacci sphere
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(num_points)
        z = np.linspace(1 - 1.0 / num_points, 1.0 / num_points - 1, num_points)
        radius_at_z = np.sqrt(1 - z * z)

        x = radius_at_z * np.cos(theta)
        y = radius_at_z * np.sin(theta)

        points = np.stack([x, y, z], axis=1) * radius

        return points
