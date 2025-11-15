"""Differentiable renderer for FoundationPose."""

import torch
import torch.nn as nn
from typing import Tuple


class DifferentiableRenderer(nn.Module):
    """Simplified differentiable renderer for pose estimation."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        render_resolution: int = 256,
    ):
        """
        Args:
            image_size: Output image size
            render_resolution: Internal rendering resolution
        """
        super(DifferentiableRenderer, self).__init__()

        self.image_size = image_size
        self.render_resolution = render_resolution

    def render(
        self,
        vertices: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render object given pose.

        Args:
            vertices: (B, N, 3) object vertices
            rotation: (B, 3, 3) rotation matrix
            translation: (B, 3) translation vector
            camera_intrinsics: (B, 3, 3) camera intrinsics

        Returns:
            rendered_image: (B, 3, H, W) rendered RGB image
        """
        # Transform vertices to camera frame
        vertices_camera = torch.bmm(vertices, rotation.transpose(1, 2)) + translation.unsqueeze(1)

        # Project to image plane
        vertices_2d = self._project_points(vertices_camera, camera_intrinsics)

        # Simplified rendering: create depth image
        rendered = self._rasterize(vertices_2d, vertices_camera[:, :, 2])

        return rendered

    def _project_points(
        self,
        points_3d: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D points to 2D image plane.

        Args:
            points_3d: (B, N, 3) 3D points in camera frame
            camera_intrinsics: (B, 3, 3) camera intrinsics

        Returns:
            points_2d: (B, N, 2) 2D projected points
        """
        # points_3d: (B, N, 3)
        # Homogeneous projection: [u, v, 1]^T = K @ [X, Y, Z]^T / Z

        fx = camera_intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = camera_intrinsics[:, 1, 1].unsqueeze(1)  # (B, 1)
        cx = camera_intrinsics[:, 0, 2].unsqueeze(1)  # (B, 1)
        cy = camera_intrinsics[:, 1, 2].unsqueeze(1)  # (B, 1)

        X = points_3d[:, :, 0]  # (B, N)
        Y = points_3d[:, :, 1]  # (B, N)
        Z = points_3d[:, :, 2] + 1e-8  # (B, N) - avoid division by zero

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        points_2d = torch.stack([u, v], dim=-1)  # (B, N, 2)

        return points_2d

    def _rasterize(
        self,
        points_2d: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified rasterization.

        Args:
            points_2d: (B, N, 2) 2D points
            depths: (B, N) depth values

        Returns:
            rendered: (B, 1, H, W) rendered depth map
        """
        B, N = points_2d.shape[:2]
        H, W = self.image_size

        # Create empty depth buffer
        depth_buffer = torch.zeros(B, H, W, device=points_2d.device)

        # Simple splatting (not differentiable, just for placeholder)
        for b in range(B):
            for i in range(N):
                u, v = points_2d[b, i]
                u_int = int(u.item())
                v_int = int(v.item())

                if 0 <= u_int < W and 0 <= v_int < H:
                    depth_buffer[b, v_int, u_int] = depths[b, i]

        return depth_buffer.unsqueeze(1)  # (B, 1, H, W)
