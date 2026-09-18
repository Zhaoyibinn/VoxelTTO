"""Build the minimal camera objects consumed by the GGGS rasterizer."""

from types import SimpleNamespace

import numpy as np
import torch


def _projection_matrix_from_k(k, width, height, znear, zfar, device):
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0] = 2.0 * fx / float(width)
    projection[1, 1] = 2.0 * fy / float(height)
    projection[0, 2] = 2.0 * (cx - (float(width) - 1.0) * 0.5) / float(width)
    projection[1, 2] = 2.0 * (cy - (float(height) - 1.0) * 0.5) / float(height)
    projection[3, 2] = 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    return projection


def build_gs_camera(K, ext, height, width, data_device="cuda"):
    """Construct one lightweight GGGS camera per batch/view."""
    batch_size, view_count = ext.shape[:2]
    intrinsics = np.asarray(K.detach().float().cpu(), dtype=np.float32).reshape(
        batch_size, view_count, 3, 3
    )
    width = int(width)
    height = int(height)
    fov_x = 2.0 * np.arctan(width / (2.0 * intrinsics[:, :, 0, 0]))
    fov_y = 2.0 * np.arctan(height / (2.0 * intrinsics[:, :, 1, 1]))
    device = torch.device(data_device)
    znear, zfar = 0.00003, 100.0

    cameras = []
    for batch_idx in range(batch_size):
        batch_cameras = []
        for view_idx in range(view_count):
            world_view = ext[batch_idx, view_idx].detach().float().to(device).transpose(0, 1)
            projection = _projection_matrix_from_k(
                intrinsics[batch_idx, view_idx], width, height, znear, zfar, device
            ).transpose(0, 1)
            full_projection = world_view.unsqueeze(0).bmm(
                projection.unsqueeze(0)
            ).squeeze(0)
            batch_cameras.append(
                SimpleNamespace(
                    image_width=width,
                    image_height=height,
                    FoVx=float(fov_x[batch_idx, view_idx]),
                    FoVy=float(fov_y[batch_idx, view_idx]),
                    znear=znear,
                    zfar=zfar,
                    world_view_transform=world_view,
                    projection_matrix=projection,
                    full_proj_transform=full_projection,
                    camera_center=torch.linalg.inv(world_view)[3, :3],
                )
            )
        cameras.append(batch_cameras)
    return cameras
