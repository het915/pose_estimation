"""Occlusion simulation utilities."""

import numpy as np
from typing import Tuple, List
import cv2


class OcclusionSimulator:
    """Simulate occlusions in images."""

    def __init__(self, occlusion_types: List[str] = ["box", "random"]):
        """
        Args:
            occlusion_types: Types of occlusions to simulate
                - 'box': Rectangular occlusions
                - 'random': Random irregular occlusions
                - 'object': Object-like occlusions
        """
        self.occlusion_types = occlusion_types

    def add_occlusion(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        target_ratio: float = 0.5,
        mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Add synthetic occlusion to images.

        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W)
            target_ratio: Target occlusion ratio [0, 1]
            mask: Optional object mask to occlude

        Returns:
            (occluded_rgb, occluded_depth, actual_ratio)
        """
        occluded_rgb = rgb.copy()
        occluded_depth = depth.copy()

        occlusion_type = np.random.choice(self.occlusion_types)

        if occlusion_type == "box":
            occluded_rgb, occluded_depth, actual_ratio = self._add_box_occlusion(
                occluded_rgb, occluded_depth, target_ratio, mask
            )
        elif occlusion_type == "random":
            occluded_rgb, occluded_depth, actual_ratio = self._add_random_occlusion(
                occluded_rgb, occluded_depth, target_ratio, mask
            )
        else:
            actual_ratio = 0.0

        return occluded_rgb, occluded_depth, actual_ratio

    def _add_box_occlusion(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        target_ratio: float,
        mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Add rectangular occlusions."""
        h, w = rgb.shape[:2]

        if mask is not None:
            # Get bounding box of object
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) == 0:
                return rgb, depth, 0.0

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            obj_h = y_max - y_min
            obj_w = x_max - x_min
            obj_area = obj_h * obj_w

            # Number of boxes to achieve target ratio
            num_boxes = max(1, int(target_ratio * 3))

            total_occluded = 0

            for _ in range(num_boxes):
                # Random box size
                box_h = np.random.randint(obj_h // 10, obj_h // 2)
                box_w = np.random.randint(obj_w // 10, obj_w // 2)

                # Random position within object bbox
                box_y = np.random.randint(y_min, max(y_min + 1, y_max - box_h))
                box_x = np.random.randint(x_min, max(x_min + 1, x_max - box_w))

                # Random color
                color = np.random.randint(0, 255, size=3)

                # Apply occlusion
                rgb[box_y:box_y+box_h, box_x:box_x+box_w] = color
                depth[box_y:box_y+box_h, box_x:box_x+box_w] = 0

                # Count occluded pixels
                occluded_mask = np.zeros_like(mask, dtype=bool)
                occluded_mask[box_y:box_y+box_h, box_x:box_x+box_w] = True
                total_occluded += np.sum(mask & occluded_mask)

            actual_ratio = total_occluded / obj_area if obj_area > 0 else 0.0

        else:
            # Random boxes without mask
            num_boxes = max(1, int(target_ratio * 5))

            for _ in range(num_boxes):
                box_h = np.random.randint(h // 10, h // 4)
                box_w = np.random.randint(w // 10, w // 4)

                box_y = np.random.randint(0, max(1, h - box_h))
                box_x = np.random.randint(0, max(1, w - box_w))

                color = np.random.randint(0, 255, size=3)

                rgb[box_y:box_y+box_h, box_x:box_x+box_w] = color
                depth[box_y:box_y+box_h, box_x:box_x+box_w] = 0

            actual_ratio = target_ratio  # Approximate

        return rgb, depth, actual_ratio

    def _add_random_occlusion(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        target_ratio: float,
        mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Add random irregular occlusions."""
        h, w = rgb.shape[:2]

        if mask is not None:
            # Get object region
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) == 0:
                return rgb, depth, 0.0

            obj_pixels = len(coords)
            target_pixels = int(obj_pixels * target_ratio)

            # Randomly select pixels to occlude
            occlude_indices = np.random.choice(
                len(coords), min(target_pixels, len(coords)), replace=False
            )
            occlude_coords = coords[occlude_indices]

            # Apply occlusion
            for y, x in occlude_coords:
                color = np.random.randint(0, 255, size=3)
                # Occlude neighborhood
                y1, y2 = max(0, y-2), min(h, y+3)
                x1, x2 = max(0, x-2), min(w, x+3)

                rgb[y1:y2, x1:x2] = color
                depth[y1:y2, x1:x2] = 0

            actual_ratio = len(occlude_indices) / obj_pixels

        else:
            # Random occlusion without mask
            num_patches = max(1, int(target_ratio * 10))

            for _ in range(num_patches):
                center_y = np.random.randint(0, h)
                center_x = np.random.randint(0, w)

                radius = np.random.randint(10, 50)

                # Create circular occlusion
                y, x = np.ogrid[:h, :w]
                circle_mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2

                color = np.random.randint(0, 255, size=3)
                rgb[circle_mask] = color
                depth[circle_mask] = 0

            actual_ratio = target_ratio  # Approximate

        return rgb, depth, actual_ratio

    def create_occlusion_mask(
        self,
        shape: Tuple[int, int],
        occlusion_ratio: float,
    ) -> np.ndarray:
        """
        Create a binary occlusion mask.

        Args:
            shape: (height, width)
            occlusion_ratio: Target occlusion ratio

        Returns:
            Binary mask (1 = occluded, 0 = visible)
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        target_pixels = int(h * w * occlusion_ratio)

        # Random box occlusions
        num_boxes = max(1, int(occlusion_ratio * 5))

        for _ in range(num_boxes):
            box_h = np.random.randint(h // 10, h // 3)
            box_w = np.random.randint(w // 10, w // 3)

            box_y = np.random.randint(0, max(1, h - box_h))
            box_x = np.random.randint(0, max(1, w - box_w))

            mask[box_y:box_y+box_h, box_x:box_x+box_w] = 1

        return mask
