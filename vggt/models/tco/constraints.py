"""Differentiable TCO prior penalties.

Camera conventions follow this repository: extrinsics are world-to-camera and
may be either 3x4 or 4x4 matrices.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def as_homogeneous(matrices: torch.Tensor) -> torch.Tensor:
    if matrices.shape[-2:] == (4, 4):
        return matrices
    if matrices.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics ending in 3x4 or 4x4, got {matrices.shape}")
    bottom = torch.zeros(*matrices.shape[:-2], 1, 4, device=matrices.device, dtype=matrices.dtype)
    bottom[..., 0, 3] = 1
    return torch.cat((matrices, bottom), dim=-2)


def _canonical_camera_poses(extrinsics: torch.Tensor) -> torch.Tensor:
    poses = torch.linalg.inv(as_homogeneous(extrinsics))
    return torch.linalg.inv(poses[:, :1]) @ poses


def camera_pose_energy(
    predicted_extrinsics: torch.Tensor,
    prior_extrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TCO pose penalty with frame and scene-scale invariance."""
    if predicted_extrinsics.ndim == 3:
        predicted_extrinsics = predicted_extrinsics.unsqueeze(0)
    if prior_extrinsics.ndim == 3:
        prior_extrinsics = prior_extrinsics.unsqueeze(0)
    predicted = _canonical_camera_poses(predicted_extrinsics.float())
    prior = _canonical_camera_poses(prior_extrinsics.to(predicted).float())

    relative_rotation = predicted[..., :3, :3].transpose(-1, -2) @ prior[..., :3, :3]
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    rotation_loss = ((3.0 - trace) * 0.5).mean()

    pred_t = predicted[..., :3, 3]
    prior_t = prior[..., :3, 3]
    pred_scale = pred_t.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-6)
    prior_scale = prior_t.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-6)
    translation_loss = F.smooth_l1_loss(
        pred_t / pred_scale.unsqueeze(-1),
        prior_t / prior_scale.unsqueeze(-1),
    )
    return rotation_loss, translation_loss


def intrinsics_energy(
    predicted: torch.Tensor,
    prior: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Penalty on fx, fy, cx and cy in normalized image coordinates."""
    if predicted.ndim == 4 and prior.ndim == 3:
        prior = prior.unsqueeze(0)
    prior = prior.to(device=predicted.device, dtype=predicted.dtype)
    if predicted.shape != prior.shape:
        raise ValueError(f"Intrinsics shape mismatch: {predicted.shape} versus {prior.shape}")
    height, width = image_hw
    normalizer = predicted.new_tensor([width, height, width, height])
    pred_values = torch.stack(
        (predicted[..., 0, 0], predicted[..., 1, 1], predicted[..., 0, 2], predicted[..., 1, 2]),
        dim=-1,
    ) / normalizer
    prior_values = torch.stack(
        (prior[..., 0, 0], prior[..., 1, 1], prior[..., 0, 2], prior[..., 1, 2]),
        dim=-1,
    ) / normalizer
    return F.smooth_l1_loss(pred_values, prior_values)


def depth_energy(
    predicted: torch.Tensor,
    prior: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scale/shift-invariant depth prior loss with per-scene alignment."""
    if predicted.ndim == 5 and predicted.shape[-1] == 1:
        predicted = predicted[..., 0]
    if prior.ndim == 5 and prior.shape[2] == 1:
        prior = prior[:, :, 0]
    elif prior.ndim == 5 and prior.shape[-1] == 1:
        prior = prior[..., 0]
    elif prior.ndim == 4 and prior.shape[1] == 1:
        prior = prior[:, 0]
    elif prior.ndim == 4 and prior.shape[-1] == 1:
        prior = prior[..., 0]
    if predicted.ndim == 4 and prior.ndim == 3:
        prior = prior.unsqueeze(0)
    if predicted.shape != prior.shape:
        raise ValueError(f"Depth shape mismatch: {predicted.shape} versus {prior.shape}")

    prior = prior.to(device=predicted.device, dtype=predicted.dtype)
    valid = torch.isfinite(predicted) & torch.isfinite(prior) & (prior > 0)
    if mask is not None:
        if mask.ndim == 5 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.ndim == 5 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape != valid.shape:
            raise ValueError(f"Depth mask shape mismatch: {mask.shape} versus {valid.shape}")
        valid &= mask.to(device=valid.device, dtype=torch.bool)

    losses = []
    for batch_index in range(predicted.shape[0]):
        x = predicted[batch_index][valid[batch_index]].float()
        y = prior[batch_index][valid[batch_index]].float()
        if x.numel() < 2:
            continue
        design = torch.stack((x, torch.ones_like(x)), dim=-1)
        # Least-squares alignment is differentiable with respect to x.
        solution = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution[:, 0]
        aligned = solution[0] * x + solution[1]
        losses.append(F.smooth_l1_loss(aligned, y))
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean().to(predicted.dtype)

