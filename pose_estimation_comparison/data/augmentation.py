"""Data augmentation for pose estimation."""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Dict


class DataAugmentation:
    """Data augmentation for 6D pose estimation."""

    def __init__(
        self,
        random_rotation: bool = True,
        random_translation: float = 0.05,
        random_scale: tuple = (0.8, 1.2),
        color_jitter: bool = True,
        lighting_variation: bool = False,
        synthetic_occlusion: bool = False,
        occlusion_prob: float = 0.5,
    ):
        """
        Args:
            random_rotation: Apply random rotations
            random_translation: Max translation in meters
            random_scale: (min, max) scale factors
            color_jitter: Apply color jittering
            lighting_variation: Simulate lighting changes
            synthetic_occlusion: Add synthetic occlusions
            occlusion_prob: Probability of applying synthetic occlusion
        """
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.random_scale = random_scale
        self.color_jitter = color_jitter
        self.lighting_variation = lighting_variation
        self.synthetic_occlusion = synthetic_occlusion
        self.occlusion_prob = occlusion_prob

    def __call__(self, data: Dict) -> Dict:
        """Apply augmentations to the data."""
        # Random rotation
        if self.random_rotation and np.random.random() > 0.5:
            data = self._apply_rotation(data)

        # Random translation
        if self.random_translation > 0 and np.random.random() > 0.5:
            data = self._apply_translation(data)

        # Random scale
        if self.random_scale and np.random.random() > 0.5:
            data = self._apply_scale(data)

        # Color jitter
        if self.color_jitter and "rgb" in data and np.random.random() > 0.5:
            data = self._apply_color_jitter(data)

        # Lighting variation
        if self.lighting_variation and "rgb" in data and np.random.random() > 0.5:
            data = self._apply_lighting(data)

        # Synthetic occlusion
        if self.synthetic_occlusion and np.random.random() < self.occlusion_prob:
            data = self._apply_synthetic_occlusion(data)

        return data

    def _apply_rotation(self, data: Dict) -> Dict:
        """Apply random rotation perturbation."""
        # Small rotation perturbation (±10 degrees)
        angle = np.random.uniform(-10, 10)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)

        rot_perturb = R.from_rotvec(np.radians(angle) * axis).as_matrix()

        # Update ground truth rotation
        data["rotation"] = rot_perturb @ data["rotation"]

        # Rotate point cloud if available
        if "point_cloud" in data:
            data["point_cloud"] = data["point_cloud"] @ rot_perturb.T

        return data

    def _apply_translation(self, data: Dict) -> Dict:
        """Apply random translation perturbation."""
        trans_perturb = np.random.uniform(
            -self.random_translation,
            self.random_translation,
            size=3
        )

        # Update ground truth translation
        data["translation"] = data["translation"] + trans_perturb

        # Translate point cloud if available
        if "point_cloud" in data:
            data["point_cloud"] = data["point_cloud"] + trans_perturb

        return data

    def _apply_scale(self, data: Dict) -> Dict:
        """Apply random scale perturbation."""
        scale = np.random.uniform(self.random_scale[0], self.random_scale[1])

        # Scale point cloud
        if "point_cloud" in data:
            data["point_cloud"] = data["point_cloud"] * scale

        # Scale translation
        data["translation"] = data["translation"] * scale

        # Scale depth
        if "depth" in data:
            data["depth"] = data["depth"] * scale

        return data

    def _apply_color_jitter(self, data: Dict) -> Dict:
        """Apply color jittering to RGB image."""
        rgb = data["rgb"]

        # Brightness
        brightness = np.random.uniform(0.7, 1.3)
        rgb = np.clip(rgb * brightness, 0, 255)

        # Contrast
        contrast = np.random.uniform(0.7, 1.3)
        mean = rgb.mean()
        rgb = np.clip((rgb - mean) * contrast + mean, 0, 255)

        # Saturation
        saturation = np.random.uniform(0.7, 1.3)
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = np.clip(gray * (1 - saturation) + rgb * saturation, 0, 255)

        data["rgb"] = rgb.astype(np.uint8)
        return data

    def _apply_lighting(self, data: Dict) -> Dict:
        """Apply lighting variation."""
        rgb = data["rgb"].astype(np.float32)

        # Add directional lighting effect
        h, w = rgb.shape[:2]
        gradient = np.linspace(0.8, 1.2, w)
        gradient = np.tile(gradient, (h, 1))
        gradient = np.expand_dims(gradient, axis=2)

        rgb = np.clip(rgb * gradient, 0, 255)

        data["rgb"] = rgb.astype(np.uint8)
        return data

    def _apply_synthetic_occlusion(self, data: Dict) -> Dict:
        """Add synthetic occlusion patches."""
        if "rgb" in data:
            rgb = data["rgb"]
            h, w = rgb.shape[:2]

            # Random rectangular occlusions
            num_occlusions = np.random.randint(1, 4)

            for _ in range(num_occlusions):
                # Random size (10-30% of image)
                occ_h = np.random.randint(int(h * 0.1), int(h * 0.3))
                occ_w = np.random.randint(int(w * 0.1), int(w * 0.3))

                # Random position
                y = np.random.randint(0, max(1, h - occ_h))
                x = np.random.randint(0, max(1, w - occ_w))

                # Random color
                color = np.random.randint(0, 255, size=3)

                # Apply occlusion
                rgb[y:y+occ_h, x:x+occ_w] = color

            data["rgb"] = rgb

        if "depth" in data:
            depth = data["depth"]
            h, w = depth.shape

            # Similar occlusion for depth
            num_occlusions = np.random.randint(1, 3)

            for _ in range(num_occlusions):
                occ_h = np.random.randint(int(h * 0.1), int(h * 0.3))
                occ_w = np.random.randint(int(w * 0.1), int(w * 0.3))

                y = np.random.randint(0, max(1, h - occ_h))
                x = np.random.randint(0, max(1, w - occ_w))

                # Zero out depth (simulating occlusion)
                depth[y:y+occ_h, x:x+occ_w] = 0

            data["depth"] = depth

        return data
