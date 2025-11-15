"""Scene generation utilities."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .pybullet_env import PyBulletEnv


class SceneGenerator:
    """Generate diverse scenes for training."""

    def __init__(
        self,
        env: PyBulletEnv,
        object_models: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            env: PyBullet environment
            object_models: Dictionary mapping object IDs to mesh paths
        """
        self.env = env
        self.object_models = object_models or {}

    def generate_random_scene(
        self,
        num_objects: int = 1,
        num_occluders: int = 0,
        workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.3, 0.3),
            (-0.3, 0.3),
            (0.5, 1.5),
        ),
    ) -> List[Dict]:
        """
        Generate a random scene with objects and occluders.

        Args:
            num_objects: Number of target objects
            num_occluders: Number of occluding objects
            workspace_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))

        Returns:
            List of object information dictionaries
        """
        self.env.reset()

        # Load plane
        self.env.load_plane()

        objects_info = []

        # Place target objects
        for i in range(num_objects):
            # Random object selection
            if self.object_models:
                obj_id = np.random.choice(list(self.object_models.keys()))
                mesh_path = self.object_models[obj_id]
            else:
                obj_id = f"obj_{i:02d}"
                mesh_path = None

            # Random pose
            position = self._sample_random_position(workspace_bounds)
            orientation = self._sample_random_orientation()

            # Load object
            body_id = self.env.load_object(
                obj_id=obj_id,
                mesh_path=mesh_path,
                position=position,
                orientation=orientation,
            )

            # Store info
            rotation, translation = self.env.get_object_pose(obj_id)

            objects_info.append({
                "object_id": obj_id,
                "body_id": body_id,
                "rotation": rotation,
                "translation": translation,
                "is_target": True,
            })

        # Place occluders
        for i in range(num_occluders):
            # Random position between camera and object
            target_obj = objects_info[0]  # Occlude first object
            target_pos = target_obj["translation"]

            # Position occluder between camera and target
            camera_pos = np.array([0, 0, 1.0])
            occluder_pos = self._sample_occluder_position(camera_pos, target_pos)

            # Random size
            size = np.random.uniform(0.05, 0.15, size=3)

            # Add occluder
            self.env.add_occluder(
                position=tuple(occluder_pos),
                size=tuple(size),
            )

        return objects_info

    def generate_scene_with_target_occlusion(
        self,
        target_occlusion: float,
        obj_id: str,
        max_attempts: int = 50,
        workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.3, 0.3),
            (-0.3, 0.3),
            (0.5, 1.5),
        ),
    ) -> Optional[Dict]:
        """
        Generate a scene with a specific occlusion level.

        Args:
            target_occlusion: Target occlusion ratio [0, 1]
            obj_id: Object ID
            max_attempts: Maximum attempts to achieve target occlusion
            workspace_bounds: Workspace bounds

        Returns:
            Scene information dictionary or None if failed
        """
        tolerance = 0.1  # ±10% tolerance

        for attempt in range(max_attempts):
            self.env.reset()
            self.env.load_plane()

            # Place target object
            position = self._sample_random_position(workspace_bounds)
            orientation = self._sample_random_orientation()

            mesh_path = self.object_models.get(obj_id)

            self.env.load_object(
                obj_id=obj_id,
                mesh_path=mesh_path,
                position=position,
                orientation=orientation,
            )

            # Add occluders iteratively
            num_occluders = max(1, int(target_occlusion * 5))  # Estimate

            for i in range(num_occluders):
                target_pos = np.array(position)
                camera_pos = np.array([0, 0, 1.0])

                occluder_pos = self._sample_occluder_position(camera_pos, target_pos)
                size = np.random.uniform(0.05, 0.15, size=3)

                self.env.add_occluder(
                    position=tuple(occluder_pos),
                    size=tuple(size),
                )

            # Check occlusion ratio
            actual_occlusion = self.env.compute_occlusion_ratio(obj_id)

            if abs(actual_occlusion - target_occlusion) <= tolerance:
                # Success
                rotation, translation = self.env.get_object_pose(obj_id)

                return {
                    "object_id": obj_id,
                    "rotation": rotation,
                    "translation": translation,
                    "target_occlusion": target_occlusion,
                    "actual_occlusion": actual_occlusion,
                }

        # Failed to achieve target occlusion
        return None

    def _sample_random_position(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        """Sample random position within bounds."""
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        z = np.random.uniform(bounds[2][0], bounds[2][1])

        return (x, y, z)

    def _sample_random_orientation(self) -> Tuple[float, float, float, float]:
        """Sample random orientation as quaternion."""
        # Random quaternion (uniform on sphere)
        u1, u2, u3 = np.random.uniform(0, 1, size=3)

        q = np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ])

        return tuple(q)

    def _sample_occluder_position(
        self,
        camera_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """Sample occluder position between camera and target."""
        # Random point along line from camera to target
        t = np.random.uniform(0.3, 0.7)  # 30-70% along the line

        occluder_pos = camera_pos + t * (target_pos - camera_pos)

        # Add some random offset
        offset = np.random.uniform(-0.1, 0.1, size=3)
        occluder_pos += offset

        return occluder_pos
