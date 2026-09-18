"""Minimal GGGS renderer used by VoxelTTO inference."""

import math

import torch
from gggs_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _optional_gaussian_attribute(gaussians, name, batch_idx, default):
    if not hasattr(gaussians, name):
        return default
    value = getattr(gaussians, name)
    value = value() if callable(value) else value
    if torch.is_tensor(value) and value.ndim >= 3:
        return value[batch_idx]
    return value


def render(
    viewpoint_camera,
    gaussians,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    batch_idx=0,
    sh_degree=0,
    gs_mode="GGGS",
):
    """Render one view with the stochastic-solid GGGS rasterizer."""
    if gs_mode is not None and str(gs_mode).upper() != "GGGS":
        raise ValueError(f"VoxelTTO only supports gs_mode='GGGS', got {gs_mode!r}")

    means3d = gaussians.means[batch_idx].float()
    opacities = gaussians.opacities[batch_idx].unsqueeze(-1).float()
    bg_color = bg_color.float()
    screenspace_points = torch.zeros_like(
        means3d, requires_grad=True, device=means3d.device
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        kernel_size=float(getattr(pipe, "kernel_size", 0.0)),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.float(),
        projmatrix=viewpoint_camera.full_proj_transform.float(),
        sh_degree=int(sh_degree),
        sg_degree=0,
        campos=viewpoint_camera.camera_center.float(),
        prefiltered=False,
        require_depth=bool(getattr(pipe, "require_depth", True)),
        debug=bool(getattr(pipe, "debug", False)),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if "cuda" not in str(means3d.device):
        return {
            "render": None,
            "viewspace_points": None,
            "visibility_filter": None,
            "radii": None,
        }

    scales = gaussians.scales[batch_idx].float()
    rotations = gaussians.rotations[batch_idx].float()
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = gaussians.harmonics[batch_idx].permute(0, 2, 1).float()
    else:
        colors_precomp = override_color.float()

    empty = torch.empty(0, dtype=means3d.dtype, device=means3d.device)
    sg_axis = _optional_gaussian_attribute(gaussians, "sg_axis", batch_idx, empty)
    sg_sharpness = _optional_gaussian_attribute(
        gaussians, "sg_sharpness", batch_idx, empty
    )
    sg_color = _optional_gaussian_attribute(gaussians, "sg_color", batch_idx, empty)
    sg_axis = empty if sg_axis is None else sg_axis.float()
    sg_sharpness = empty if sg_sharpness is None else sg_sharpness.float()
    sg_color = empty if sg_color is None else sg_color.float()

    rendered_image, radii, median_depth, alpha, normal = rasterizer(
        means3D=means3d,
        means2D=screenspace_points,
        shs=shs,
        sg_axis=sg_axis,
        sg_sharpness=sg_sharpness,
        sg_color=sg_color,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
    )
    return {
        "render": rendered_image,
        "depth": median_depth,
        "alpha": alpha,
        "median_depth": median_depth,
        "normal": normal,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
