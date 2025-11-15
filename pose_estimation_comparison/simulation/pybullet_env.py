"""PyBullet environment for realistic rendering and physics simulation."""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Optional, Dict
import cv2


class PyBulletEnv:
    """PyBullet environment for pose estimation data generation."""

    def __init__(
        self,
        gui: bool = False,
        image_size: Tuple[int, int] = (640, 480),
        camera_distance: float = 1.0,
    ):
        """
        Args:
            gui: Whether to use GUI mode
            image_size: (width, height) of rendered images
            camera_distance: Default camera distance from origin
        """
        self.gui = gui
        self.image_size = image_size
        self.camera_distance = camera_distance

        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Set up data path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physics settings
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Camera parameters
        self.camera_intrinsics = self._compute_camera_intrinsics()

        # Object IDs
        self.object_ids = {}

    def _compute_camera_intrinsics(self) -> np.ndarray:
        """Compute camera intrinsic matrix."""
        width, height = self.image_size

        # Field of view
        fov = 60  # degrees
        aspect = width / height

        # Compute focal length
        fy = height / (2.0 * np.tan(np.radians(fov) / 2.0))
        fx = fy

        # Principal point
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return intrinsics

    def reset(self):
        """Reset the environment."""
        # Remove all objects
        for obj_id in self.object_ids.values():
            p.removeBody(obj_id)

        self.object_ids = {}

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

    def load_plane(self) -> int:
        """Load ground plane."""
        plane_id = p.loadURDF("plane.urdf")
        self.object_ids["plane"] = plane_id
        return plane_id

    def load_object(
        self,
        obj_id: str,
        mesh_path: Optional[str] = None,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        scale: float = 1.0,
    ) -> int:
        """
        Load an object into the scene.

        Args:
            obj_id: Object identifier
            mesh_path: Path to mesh file (OBJ or URDF)
            position: (x, y, z) position
            orientation: (x, y, z, w) quaternion orientation
            scale: Object scale

        Returns:
            PyBullet body ID
        """
        if mesh_path is None:
            # Create primitive shape
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[np.random.random(), np.random.random(), np.random.random(), 1],
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
            )

            body_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation,
            )
        else:
            # Load from file
            if mesh_path.endswith('.urdf'):
                body_id = p.loadURDF(
                    mesh_path,
                    basePosition=position,
                    baseOrientation=orientation,
                    globalScaling=scale,
                )
            else:
                # Load mesh (OBJ)
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=mesh_path,
                    meshScale=[scale, scale, scale],
                )
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=mesh_path,
                    meshScale=[scale, scale, scale],
                )

                body_id = p.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    baseOrientation=orientation,
                )

        self.object_ids[obj_id] = body_id
        return body_id

    def add_occluder(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    ) -> int:
        """Add an occluding object to the scene."""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=size,
            rgbaColor=[0.5, 0.5, 0.5, 1],
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=size,
        )

        occluder_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )

        occluder_name = f"occluder_{len([k for k in self.object_ids if k.startswith('occluder')])}"
        self.object_ids[occluder_name] = occluder_id

        return occluder_id

    def get_object_pose(self, obj_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get object pose.

        Returns:
            (rotation_matrix, translation_vector)
        """
        if obj_id not in self.object_ids:
            raise ValueError(f"Object {obj_id} not found")

        body_id = self.object_ids[obj_id]
        position, orientation = p.getBasePositionAndOrientation(body_id)

        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        translation = np.array(position)

        return rotation_matrix, translation

    def set_object_pose(
        self,
        obj_id: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ):
        """Set object pose."""
        if obj_id not in self.object_ids:
            raise ValueError(f"Object {obj_id} not found")

        body_id = self.object_ids[obj_id]
        p.resetBasePositionAndOrientation(body_id, position, orientation)

    def render(
        self,
        camera_position: Optional[Tuple[float, float, float]] = None,
        target_position: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Render RGB and depth images.

        Returns:
            Dictionary with 'rgb' and 'depth' arrays
        """
        if camera_position is None:
            camera_position = (0, 0, self.camera_distance)

        if target_position is None:
            target_position = (0, 0, 0)

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1],
        )

        # Compute projection matrix
        width, height = self.image_size
        fov = 60
        aspect = width / height
        near = 0.01
        far = 10.0

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
        )

        # Render
        width, height, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
        )

        # Process RGB
        rgb_array = np.array(rgb).reshape(height, width, 4)[:, :, :3]

        # Process depth
        depth_buffer = np.array(depth).reshape(height, width)
        depth_array = far * near / (far - (far - near) * depth_buffer)

        return {
            "rgb": rgb_array,
            "depth": depth_array,
            "segmentation": np.array(seg).reshape(height, width),
        }

    def get_segmentation_mask(self, obj_id: str) -> np.ndarray:
        """Get segmentation mask for a specific object."""
        render_output = self.render()
        seg = render_output["segmentation"]

        if obj_id not in self.object_ids:
            return np.zeros_like(seg, dtype=bool)

        body_id = self.object_ids[obj_id]
        mask = (seg == body_id)

        return mask

    def compute_occlusion_ratio(self, obj_id: str) -> float:
        """
        Compute occlusion ratio for an object.

        Returns:
            Occlusion ratio [0, 1]
        """
        # Render with and without occluders
        render_output = self.render()
        seg_with_occlusion = render_output["segmentation"]

        # Remove occluders temporarily
        occluder_ids = []
        for name, body_id in list(self.object_ids.items()):
            if name.startswith("occluder"):
                occluder_ids.append((name, p.getBasePositionAndOrientation(body_id)))
                p.removeBody(body_id)
                del self.object_ids[name]

        # Render without occlusion
        render_output_no_occ = self.render()
        seg_no_occlusion = render_output_no_occ["segmentation"]

        # Restore occluders
        for name, (pos, ori) in occluder_ids:
            self.add_occluder(pos)

        # Compute occlusion
        if obj_id not in self.object_ids:
            return 0.0

        body_id = self.object_ids[obj_id]

        visible_pixels = np.sum(seg_with_occlusion == body_id)
        total_pixels = np.sum(seg_no_occlusion == body_id)

        if total_pixels == 0:
            return 1.0  # Fully occluded

        occlusion_ratio = 1.0 - (visible_pixels / total_pixels)

        return float(np.clip(occlusion_ratio, 0.0, 1.0))

    def close(self):
        """Close the environment."""
        p.disconnect()
