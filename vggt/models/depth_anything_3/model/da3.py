# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from vggt.models.depth_anything_3.cfg import create_object
from vggt.models.depth_anything_3.specs import Gaussians
from vggt.models.depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from vggt.models.depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from vggt.models.depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, unproject_depth



class RecoverableVoxelLimitError(RuntimeError):
    """Signals that a training scene is too large for the Gaussian decoder."""

    recoverable_training_batch = True

    def __init__(self, voxel_counts: list[int], threshold: int) -> None:
        self.voxel_counts = voxel_counts
        self.threshold = threshold
        super().__init__(
            f"voxel limit exceeded: counts={voxel_counts}, threshold={threshold}"
        )


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: voxel-decoded Gaussian splats in world space
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, backend=None, backend_dense_head=None, backend_gs_decoder=None,
                 backend_option=None,
                 **kwargs):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.backend = None
        if backend is not None:
            self.backend = backend if isinstance(backend, nn.Module) else create_object(_wrap_cfg(backend))

        self.backend_dense_head = None
        if backend_dense_head is not None:
            self.backend_dense_head = (
                backend_dense_head
                if isinstance(backend_dense_head, nn.Module)
                else create_object(_wrap_cfg(backend_dense_head))
            )
            
        self.backend_gs_decoder = None
        if backend_gs_decoder is not None:
            self.backend_gs_decoder = backend_gs_decoder if isinstance(backend_gs_decoder, nn.Module) else create_object(_wrap_cfg(backend_gs_decoder))

        self.backend_dense_feature_proj = None
        dense_feature_dim = self._get_backend_dense_feature_dim()
        if self.backend_gs_decoder is not None and dense_feature_dim is not None:
            if dense_feature_dim == self.backend_gs_decoder.out_dim:
                self.backend_dense_feature_proj = nn.Identity()
            else:
                self.backend_dense_feature_proj = nn.Linear(dense_feature_dim, self.backend_gs_decoder.out_dim)

        _opt = backend_option or {}
        self.high_resolution_backend_uptimes = _opt.get("high_resolution_uptimes", 6)
        self.high_resolution_backend_voxelsize = _opt.get("high_resolution_voxelsize", 0.0025)
        if int(_opt.get("iter", 0)) != 0:
            raise ValueError("VoxelTTO supports only backend_option.iter=0")
        self.backend_max_voxels = _opt.get("max_voxels", 5_000_000)
        self.skip_voxel_threshold = _opt.get("skip_voxel_threshold")
        if self.skip_voxel_threshold is not None:
            self.skip_voxel_threshold = int(self.skip_voxel_threshold)
            if self.skip_voxel_threshold <= 0:
                raise ValueError("skip_voxel_threshold must be greater than zero")
   
   
    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        rgb_images: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        # iters: int = 1,
        gs_from_backend: bool = False,
        metric_output: Dict[str, torch.Tensor] | None = None,
        valid_image_mask: torch.Tensor | None = None,
        force_input_camera: bool = False,
        return_backend_voxels: bool = False,
        timing_callback=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) 
            intrinsics: Camera intrinsics (B, N, 3, 3) 
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            ref_view_strategy: Strategy for selecting reference view
            metric_output: Pre-computed metric branch output for metric-scale pre-scaling before backend

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        if timing_callback is not None:
            timing_callback("inference/backbone", True)
        try:
            feats, aux_feats = self.backbone(
                x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
            )
        finally:
            if timing_callback is not None:
                timing_callback("inference/backbone", False)
        if not self.training:
            del cam_token
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]
        if valid_image_mask is not None:
            expected_mask_shape = x.shape[:2] + (H, W)
            if tuple(valid_image_mask.shape) != tuple(expected_mask_shape):
                raise ValueError(
                    "valid_image_mask must have shape "
                    f"{tuple(expected_mask_shape)}, got {tuple(valid_image_mask.shape)}"
                )
            valid_image_mask = valid_image_mask.to(device=x.device, dtype=torch.bool)

        # iter=0 performs exactly one frontend pass followed by the final
        # high-resolution voxel aggregation used by the Gaussian decoder.
        last_backend_voxel_detail = None
        metric_scale_factor = None

        high_resolution_backend_uptimes = self.high_resolution_backend_uptimes
        high_resolution_backend_voxelsize = self.high_resolution_backend_voxelsize
    
        if timing_callback is not None:
            timing_callback("inference/depth_camera_decoder", True)
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                output = self._process_depth_head(feats, H, W)
                output = self._process_camera_estimation(feats, H, W, output)
            if force_input_camera and extrinsics is not None and intrinsics is not None:
                output.extrinsics = extrinsics
                output.intrinsics = intrinsics
            if valid_image_mask is not None and output.get("depth_conf", None) is not None:
                confidence_mask = valid_image_mask
                while confidence_mask.ndim < output.depth_conf.ndim:
                    confidence_mask = confidence_mask.unsqueeze(-1)
                output.depth_conf = output.depth_conf.masked_fill(
                    ~confidence_mask, -torch.inf
                )
        finally:
            if timing_callback is not None:
                timing_callback("inference/depth_camera_decoder", False)

        # Before backend: apply metric scale so voxel partitioning uses metric-scale world points.
        # Scale factor is computed once from the first frontend pass and reused in subsequent passes.
        if metric_output is not None:
            if metric_scale_factor is None:
                metric_scale_factor = self._compute_metric_scale_factor(output, metric_output)
            if metric_scale_factor is not None:
                output = self._apply_metric_scale_to_output(output, metric_scale_factor)
        output.depth = output.depth.clamp(min=0.01, max=10.0)

        if getattr(self, "backend", None) is not None:
            # Final high-resolution voxel features consumed by backend GS.
            if timing_callback is not None:
                timing_callback("inference/voxel_features", True)

            upsample_factor = high_resolution_backend_uptimes
            voxel_size = high_resolution_backend_voxelsize
            pts_resized, feat_resized, shape_ctx = self._prepare_backend_points_and_features(
                feats=feats,
                output=output,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                H=H,
                W=W,
                upsample_factor = upsample_factor,
            )
            # 如果打开了bf16，那么特征是bf16的，三维点永远是float的
            


            
            # Fetch RGB colors and frozen dense image descriptors at the
            # exact same sampling locations as backend points.
            colors_resized = None
            if rgb_images is not None:
                # rgb_images shape is typically (B, N, 3, H, W)
                colors_resized = self._resize_bt_hwc(
                    rgb_images.permute(0, 1, 3, 4, 2), # from (B, N, 3, H, W) to (B, N, H, W, 3) 
                    target_size=(shape_ctx[0] * upsample_factor, shape_ctx[1] * upsample_factor) # corresponding to ph*2, pw*2
                )

            image_features_resized = None
            if image_features is not None:
                image_features_resized = self._resize_bt_hwc(
                    image_features,
                    target_size=(shape_ctx[0] * upsample_factor, shape_ctx[1] * upsample_factor),
                )
                image_features_resized = F.normalize(image_features_resized.float(), dim=-1)

            depth_conf_resized = None
            if not self.training:
                pts_resized, feat_resized, colors_resized, image_features_resized, depth_conf_resized = self._filter_backend_inputs_by_confidence(
                    pts_resized=pts_resized,
                    feat_resized=feat_resized,
                    colors_resized=colors_resized,
                    image_features_resized=image_features_resized,
                    depth_conf=output.get("depth_conf", None),
                    target_size=(shape_ctx[0] * upsample_factor, shape_ctx[1] * upsample_factor),
                    min_conf=0.0,
                )
            elif output.get("depth_conf", None) is not None:
                depth_conf_resized = self._resize_bt_hwc_min(
                    output["depth_conf"].unsqueeze(-1),
                    target_size=(shape_ctx[0] * upsample_factor, shape_ctx[1] * upsample_factor),
                )
            # output["voxel_depth_conf"] = depth_conf_resized
            voxel_details = None

            # Backend sparse voxel ops are numerically unstable in autocast.
            # Keep this path in fp32 even if outer callers enable AMP.
            with torch.amp.autocast("cuda", enabled=False):
                _, voxel_details = self.backend.forward(
                    pts_resized,
                    feat_resized,
                    voxel_sizes=[voxel_size],
                    return_voxel_details=True,
                    colors=colors_resized,
                    image_features=image_features_resized,
                    depth_conf=depth_conf_resized,
                    skip_interpolation=True,
                    chunk_size=50000,
                    timing_callback=timing_callback,
                )
            if timing_callback is not None:
                timing_callback("inference/voxel_features", False)

            if voxel_details:
                last_voxel_detail = voxel_details[-1]
                voxel_centers_list = last_voxel_detail.get("voxel_centers", [])
                total_voxels = sum(
                    centers_b.shape[0] for centers_b in voxel_centers_list if centers_b is not None
                )
                if isinstance(pts_resized, list):
                    total_points = sum(points_b.shape[0] for points_b in pts_resized)
                else:
                    total_points = pts_resized.shape[0] * pts_resized.shape[1] * pts_resized.shape[2] * pts_resized.shape[3]
                output["highres_backend_voxel_point_ratio"] = output["depth"].new_tensor(
                    float(total_voxels) / max(float(total_points), 1.0)
                )

            if not self.training:
                del pts_resized, feat_resized, colors_resized, image_features_resized, depth_conf_resized

            if gs_from_backend and voxel_details:
                voxel_centers = voxel_details[-1].get("voxel_centers", [])
                voxel_counts = [
                    int(centers.shape[0])
                    for centers in voxel_centers
                    if centers is not None
                ]
                threshold = self.skip_voxel_threshold
                if (
                    self.training
                    and threshold is not None
                    and any(count > threshold for count in voxel_counts)
                ):
                    raise RecoverableVoxelLimitError(voxel_counts, threshold)
                last_backend_voxel_detail = self._filter_backend_voxels_by_confidence(
                    voxel_details[-1],
                    max_voxels=self.backend_max_voxels,
                )
        
        with torch.autocast(device_type=x.device.type, enabled=False):
            if infer_gs:
                if not gs_from_backend:
                    raise ValueError("VoxelTTO only supports Gaussian decoding from backend voxels")
                if timing_callback is not None:
                    timing_callback("inference/gaussian_decoder", True)
                output = self.backend_gs_decoder(
                    output=output,
                    last_backend_voxel_detail=last_backend_voxel_detail,
                )
                if timing_callback is not None:
                    timing_callback("inference/gaussian_decoder", False)

        # Keep a compact description of the exact final voxels consumed by the
        # Gaussian decoder. Features are deliberately excluded because they can
        # be several GB for a large scene and are unnecessary for visualization.
        if return_backend_voxels and last_backend_voxel_detail is not None:
            output["backend_voxels"] = {
                key: last_backend_voxel_detail.get(key, None)
                for key in (
                    "voxel_centers",
                    "voxel_grid_coords",
                    "voxel_batch_ids",
                    "voxel_colors",
                    "voxel_depth_conf",
                    "voxel_size",
                )
            }
        
        output = self._process_mono_sky_estimation(output)    

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)
        return output

    def _compute_metric_scale_factor(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Compute a metric scale factor from the metric branch output.

        Must be called after the frontend pass so that output.intrinsics,
        output.depth and output.depth_conf are all available.
        Modifies metric_output.depth in-place (applies intrinsic-based normalization).
        Returns the scale factor tensor, or None if there are insufficient non-sky pixels.
        """
        if "sky" not in metric_output:
            return None

        # Normalise metric depth using predicted camera intrinsics (focal-length scaling)
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )

        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return None

        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        return least_squares_scale_scalar(valid_metric_depth, valid_depth)

    def _apply_metric_scale_to_output(
        self,
        output: Dict[str, torch.Tensor],
        scale_factor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Scale depth and extrinsics translation by scale_factor (out-of-place)."""
        output.depth = output.depth * scale_factor

        rot = output.extrinsics[:, :, :3, :3]
        trans = output.extrinsics[:, :, :3, 3] * scale_factor
        top_3 = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)
        bottom_1 = output.extrinsics[:, :, 3:, :]
        output.extrinsics = torch.cat([top_3, bottom_1], dim=-2)

        output.is_metric = 1
        output.scale_factor = scale_factor.item()
        return output

    def _resize_bt_hwc(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        batch_size, views, _, _, channels = tensor.shape
        tensor_resized = F.interpolate(
            tensor.permute(0, 1, 4, 2, 3).flatten(0, 1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        tensor_resized = tensor_resized.permute(0, 2, 3, 1).view(
            batch_size,
            views,
            target_size[0],
            target_size[1],
            channels,
        )
        return tensor_resized

    def _resize_bt_hwc_min(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        batch_size, views, height, width, channels = tensor.shape
        if (height, width) == target_size:
            return tensor

        tensor_2d = tensor.permute(0, 1, 4, 2, 3).flatten(0, 1)
        if target_size[0] <= height and target_size[1] <= width:
            tensor_resized = -F.adaptive_max_pool2d(-tensor_2d, output_size=target_size)
        else:
            tensor_resized = F.interpolate(
                tensor_2d,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        tensor_resized = tensor_resized.permute(0, 2, 3, 1).view(
            batch_size,
            views,
            target_size[0],
            target_size[1],
            channels,
        )
        return tensor_resized

    def _resize_bt_hw_mask(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        batch_size, views, _, _ = tensor.shape
        invalid_mask = (~tensor.bool()).flatten(0, 1).unsqueeze(1).float()
        invalid_mask_resized = F.adaptive_max_pool2d(invalid_mask, output_size=target_size)
        return invalid_mask_resized.view(batch_size, views, target_size[0], target_size[1]) == 0

    def _filter_backend_inputs_by_confidence(
        self,
        pts_resized: torch.Tensor,
        feat_resized: torch.Tensor,
        colors_resized: torch.Tensor | None,
        image_features_resized: torch.Tensor | None,
        depth_conf: torch.Tensor | None,
        target_size: tuple[int, int],
        min_conf: float = 5.0,
    ) -> tuple[
        torch.Tensor | list[torch.Tensor],
        torch.Tensor | list[torch.Tensor],
        torch.Tensor | list[torch.Tensor] | None,
        torch.Tensor | list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        if depth_conf is None:
            return pts_resized, feat_resized, colors_resized, image_features_resized, None

        conf_keep_mask = self._resize_bt_hw_mask(depth_conf >= min_conf, target_size)
        conf_values = self._resize_bt_hwc_min(depth_conf.unsqueeze(-1), target_size)

        filtered_pts = []
        filtered_feats = []
        filtered_colors = [] if colors_resized is not None else None
        filtered_image_features = [] if image_features_resized is not None else None
        filtered_conf = []

        for batch_idx in range(pts_resized.shape[0]):
            keep_flat = conf_keep_mask[batch_idx].reshape(-1)
            filtered_pts.append(pts_resized[batch_idx].reshape(-1, pts_resized.shape[-1])[keep_flat])
            filtered_feats.append(feat_resized[batch_idx].reshape(-1, feat_resized.shape[-1])[keep_flat])
            filtered_conf.append(conf_values[batch_idx].reshape(-1, conf_values.shape[-1])[keep_flat])
            if filtered_colors is not None:
                filtered_colors.append(colors_resized[batch_idx].reshape(-1, colors_resized.shape[-1])[keep_flat])
            if filtered_image_features is not None:
                filtered_image_features.append(
                    image_features_resized[batch_idx].reshape(-1, image_features_resized.shape[-1])[keep_flat]
                )

        return filtered_pts, filtered_feats, filtered_colors, filtered_image_features, filtered_conf

    def _prepare_backend_points_and_features(
        self,
        feats: list[torch.Tensor],
        output: Dict[str, torch.Tensor],
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        H: int,
        W: int,
        upsample_factor: int = 2,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[int, int, int, int, int] | None]:
        if self.backend is None:
            return None, None, None

        depth_us = output["depth"].unsqueeze(-1)
        # depth_min, depth_max = 0.01, 15.0
        # depth_us = depth_us.clamp(min=depth_min, max=depth_max)
        
        ctx_extr = output.get("extrinsics", extrinsics)
        ctx_intr = output.get("intrinsics", intrinsics)
        if ctx_extr is None or ctx_intr is None:
            return None, None, None

        ctx_c2w = affine_inverse(ctx_extr)
        world_points = unproject_depth(depth_us, ctx_intr, c2w=ctx_c2w)

        ph, pw = H // self.PATCH_SIZE, W // self.PATCH_SIZE
        batch_size, views, token_count, channels = feats[-1][0].shape

        backend_feat = feats[-1][0]
        if token_count > ph * pw:
            backend_feat = backend_feat[:, :, -ph * pw:, :]

        backend_feat_reshaped = backend_feat.reshape(batch_size, views, ph, pw, channels)
        target_size = (ph * upsample_factor, pw * upsample_factor)

        pts_resized = self._resize_bt_hwc(world_points, target_size=target_size)

        use_dense_head_features = (
            upsample_factor == self.PATCH_SIZE
            and self.backend_dense_feature_proj is not None
        )
        if use_dense_head_features:
            with torch.cuda.amp.autocast(enabled=False):
                dense_feat = self.backend_dense_head(feats,H,W,patch_start_idx=0,chunk_size=8)
                # dense_feat = self._extract_head_dense_features(feats, H, W, patch_start_idx=0)
                # if dense_feat.shape[2:4] != target_size:
                #     dense_feat = self._resize_bt_hwc(dense_feat, target_size=target_size)
                feat_resized = self.backend_dense_feature_proj(dense_feat)
        else:
            aligned_feat = self.backend.aligner(backend_feat_reshaped)
            feat_resized = self._resize_bt_hwc(aligned_feat, target_size=target_size)
        
        return pts_resized, feat_resized, (ph, pw, batch_size, views, token_count)

    def _get_backend_dense_feature_dim(self) -> int | None:
        if self.backend_dense_head is None or not hasattr(self.backend_dense_head, "scratch"):
            return None
        output_conv1 = getattr(self.backend_dense_head.scratch, "output_conv1", None)
        return getattr(output_conv1, "out_channels", None)

    def _filter_backend_voxels_by_confidence(
        self,
        voxel_detail: Dict[str, list[torch.Tensor] | None],
        max_voxels: int,
    ) -> Dict[str, list[torch.Tensor] | None]:
        voxel_centers = voxel_detail.get("voxel_centers", None)
        voxel_depth_conf = voxel_detail.get("voxel_depth_conf", None)
        if voxel_centers is None or voxel_depth_conf is None:
            return voxel_detail

        filtered_detail = dict(voxel_detail)
        keys_to_filter = [
            "voxel_feat",
            "voxel_centers",
            "voxel_grid_coords",
            "voxel_batch_ids",
            "voxel_colors",
            "voxel_image_features",
            "voxel_depth_conf",
        ]
        filtered_lists = {key: [] for key in keys_to_filter}

        for batch_idx, centers_b in enumerate(voxel_centers):
            conf_b = voxel_depth_conf[batch_idx] if batch_idx < len(voxel_depth_conf) else None
            if conf_b is None or centers_b is None or centers_b.shape[0] <= max_voxels:
                keep_idx = None
            else:
                conf_score = conf_b
                if conf_score.ndim > 1:
                    conf_score = conf_score.squeeze(-1)
                if conf_score.ndim > 1:
                    conf_score = conf_score.mean(dim=-1)
                keep_idx = torch.topk(conf_score, k=max_voxels, largest=True, sorted=False).indices

            for key in keys_to_filter:
                value_list = voxel_detail.get(key, None)
                if value_list is None:
                    filtered_lists[key] = None
                    continue

                value_b = value_list[batch_idx]
                if keep_idx is None or value_b is None:
                    filtered_lists[key].append(value_b)
                else:
                    filtered_lists[key].append(value_b[keep_idx])

        for key, value in filtered_lists.items():
            filtered_detail[key] = value
        return filtered_detail

    def _extract_head_dense_features(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        if self.backend_dense_head is None:
            raise RuntimeError("backend_dense_head is required to extract dense backend features.")
        return self.backend_dense_head(
            feats,
            H,
            W,
            patch_start_idx=patch_start_idx,
            chunk_size=chunk_size,
        )

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process mono sky estimation."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output
        
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        rgb_images: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        iters: int = 1,
        gs_from_backend: bool = False,
        force_input_camera: bool = False,
        valid_image_mask: torch.Tensor | None = None,
        return_backend_voxels: bool = False,
    ) -> Dict[str, torch.Tensor]:


        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing aligned depth predictions and camera parameter
s                                                                                       """
        # Run metric branch first so metric_output is ready for pre-scaling inside da3
        metric_output = self.da3_metric(x)

        # Run main branch; metric_output is passed in so the scene is scaled to metric
        # units BEFORE backend voxel partitioning (inside DepthAnything3Net.forward).
        output = self.da3(
            x,
            extrinsics,
            intrinsics,
            export_feat_layers=export_feat_layers,
            infer_gs=infer_gs,
            gs_from_backend=gs_from_backend,
            ref_view_strategy=ref_view_strategy,
            iters=iters,
            rgb_images=rgb_images,
            image_features=image_features,
            metric_output=metric_output,
            force_input_camera=force_input_camera,
            valid_image_mask=valid_image_mask,
            return_backend_voxels=return_backend_voxels,
        )

        # Depth/extrinsics are already metric-scaled inside da3 (pre-backend).
        # Only sky-region post-processing remains.
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth = output.depth * scale_factor
        
        rot = output.extrinsics[:, :, :3, :3]
        trans = output.extrinsics[:, :, :3, 3] * scale_factor
        top_3 = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)
        # if output.extrinsics.shape[-2] == 4:
        bottom_1 = output.extrinsics[:, :, 3:, :]
        output.extrinsics = torch.cat([top_3, bottom_1], dim=-2)
        # else:
        #     output.extrinsics = top_3
        
        gaussians = getattr(output, "gaussians", None)
        if gaussians is not None:
            means = getattr(gaussians, "means", None)
            scales = getattr(gaussians, "scales", None)

            if torch.is_tensor(means):
                gaussians.means = means * scale_factor
            elif isinstance(gaussians, dict) and torch.is_tensor(gaussians.get("means", None)):
                gaussians["means"] = gaussians["means"] * scale_factor

            if torch.is_tensor(scales):
                gaussians.scales = scales * scale_factor
            elif isinstance(gaussians, dict) and torch.is_tensor(gaussians.get("scales", None)):
                gaussians["scales"] = gaussians["scales"] * scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
