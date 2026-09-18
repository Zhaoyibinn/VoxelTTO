import copy
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file
from wcmatch import fnmatch

from vggt.models.depth_anything_3.cfg import create_object
from vggt.utils.specs import Gaussians
from vggt.utils.pose_enc import extri_intri_to_pose_encoding




from vggt.models.GS.utils.build_camera import build_gs_camera  
from vggt.models.GS.gaussian_renderer import render


class DepthAnything3(nn.Module):
    DEFAULT_PRETRAIN_DIRS = [
        "da3_streaming/weights_gaint_large_1.1",
        "/home/zhaoyibin/3DRE/MVS/Depth-Anything-3/da3_streaming/weights_gaint_large_1.1",
    ]
    _GS_OPS_CACHE = None
    _GLOB_FLAGS = (
        fnmatch.CASE
        | fnmatch.DOTMATCH
        | fnmatch.EXTMATCH
        | fnmatch.SPLIT
    )

    def __init__(
        self,
        model_name: str = "da3nested-giant-large",
        infer_gs: bool = False,
        gs_from_backend: bool = False,
        use_manual_metric_scaling: bool = False,
        manual_metric_divisor: float = 300.0,
        config: Dict[str, Any] | None = None,
        pretrained_dir: str | None = None,
        config_path: str | None = None,
        pretrained_weight: str | None = None,
        weights_path: str | None = None,
        load_pretrained: bool = True,
        skip_load_module_names: list[str] | None = None,
        gs_mode: str = "GGGS",
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.infer_gs = infer_gs
        self.gs_from_backend = gs_from_backend
        self.use_manual_metric_scaling = use_manual_metric_scaling
        self.manual_metric_divisor = float(manual_metric_divisor)
        self.gs_mode = str(gs_mode).upper()
        if self.gs_mode != "GGGS":
            raise ValueError(f"VoxelTTO only supports gs_mode='GGGS', got {gs_mode!r}")
        self.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        if config is None:
            if config_path is None:
                raise ValueError("DepthAnything3 requires `config` or `config_path`; no implicit fallback to pretrained_dir/config.json")
            resolved_config_path = config_path
            with open(resolved_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            config = loaded.get("config", loaded)

        config = self._rewrite_object_paths(copy.deepcopy(config))
        self.model = create_object(OmegaConf.create(config))


        if load_pretrained:
            resolved_weights_path = pretrained_weight or weights_path
            if resolved_weights_path is None:
                resolved_pretrained_dir = self._resolve_pretrained_dir(pretrained_dir)
                resolved_weights_path = os.path.join(resolved_pretrained_dir, "model.safetensors")
            
            if resolved_weights_path.endswith('.safetensors'):
                state_dict = load_file(resolved_weights_path)
            else:
                state_dict = torch.load(resolved_weights_path, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            
            if skip_load_module_names:
                self._assert_skip_module_patterns_exist(skip_load_module_names)
                state_dict, dropped_keys = self._filter_state_dict_by_module_patterns(
                    state_dict,
                    skip_load_module_names,
                )
                logging.info(
                    "Skip loading %d tensors for module patterns: %s",
                    dropped_keys,
                    skip_load_module_names,
                )
            state_dict, mismatched_keys = self._filter_state_dict_by_shape(state_dict)
            if mismatched_keys:
                logging.warning(
                    "Skip loading %d tensors due to shape mismatch: %s",
                    len(mismatched_keys),
                    mismatched_keys,
                )
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.info("Missing keys when loading pretrained weights: %s", missing_keys)
            if unexpected_keys:
                logging.info("Unexpected keys when loading pretrained weights: %s", unexpected_keys)

    def to(self, *args, **kwargs):
        # TODO: this won't work if the module is inside another module
        self.model = self.model.to(*args, **kwargs)
        self._imagenet_mean = self._imagenet_mean.to(*args, **kwargs)
        self._imagenet_std = self._imagenet_std.to(*args, **kwargs)

        # VGGT-X-style DPT precision boundary. ``self.model.to(dtype)`` casts
        # every child recursively, so restore only the final dense prediction
        # heads to FP32 afterwards. The transformer and DPT fusion features
        # remain in the requested dtype (for example BF16).
        dpt_head = getattr(self.model, "head", None)
        scratch = getattr(dpt_head, "scratch", None)
        if scratch is not None:
            output_conv2 = getattr(scratch, "output_conv2", None)
            if output_conv2 is not None:
                output_conv2.float()

            output_conv2_aux = getattr(scratch, "output_conv2_aux", None)
            if output_conv2_aux is not None:
                output_conv2_aux.float()

        # Camera encoding/decoding is small compared with the image backbone.
        # Keep both complete parameter/activation paths in FP32 for camera
        # conditioning, pose regression, and camera-matrix construction.
        camera_enc = getattr(self.model, "cam_enc", None)
        if camera_enc is not None:
            camera_enc.float()

        camera_dec = getattr(self.model, "cam_dec", None)
        if camera_dec is not None:
            camera_dec.float()

        # Run SpUNet and the final Gaussian decoder in FP32.
        gaussian_decoder = getattr(self.model, "backend_gs_decoder", None)
        if gaussian_decoder is not None:
            gaussian_decoder.float()

        return self

    def _assert_skip_module_patterns_exist(self, patterns: list[str]) -> None:
        module_names = [name for name, _ in self.named_modules()]
        not_found = [
            p
            for p in patterns
            if not any(fnmatch.fnmatch(name, p, flags=self._GLOB_FLAGS) for name in module_names)
        ]
        assert not not_found, f"These skip_load_module_names patterns matched no modules: {not_found}"

    def _filter_state_dict_by_module_patterns(
        self,
        state_dict: Dict[str, torch.Tensor],
        patterns: list[str],
    ):
        filtered = {}
        dropped = 0
        for key, value in state_dict.items():
            should_drop = False
            for pattern in patterns:
                if fnmatch.fnmatch(key, pattern, flags=self._GLOB_FLAGS) or fnmatch.fnmatch(
                    key,
                    f"{pattern}.*",
                    flags=self._GLOB_FLAGS,
                ):
                    should_drop = True
                    break

            if should_drop:
                dropped += 1
                continue

            filtered[key] = value

        return filtered, dropped

    def _filter_state_dict_by_shape(self, state_dict: Dict[str, torch.Tensor]):
        current_state_dict = self.state_dict()
        filtered = {}
        mismatched_keys = []

        for key, value in state_dict.items():
            current_value = current_state_dict.get(key)
            if current_value is not None and torch.is_tensor(value) and torch.is_tensor(current_value):
                if value.shape != current_value.shape:
                    mismatched_keys.append(
                        f"{key}: checkpoint {tuple(value.shape)} != model {tuple(current_value.shape)}"
                    )
                    continue

            filtered[key] = value

        return filtered, mismatched_keys

    def _resolve_pretrained_dir(self, pretrained_dir: str | None) -> str:
        if pretrained_dir is not None:
            return pretrained_dir

        for candidate in self.DEFAULT_PRETRAIN_DIRS:
            if os.path.exists(candidate):
                return candidate

        return self.DEFAULT_PRETRAIN_DIRS[0]

    def _rewrite_object_paths(self, cfg: Any) -> Any:
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            out = {}
            for key, value in cfg.items():
                if key == "__object__" and isinstance(value, dict) and "path" in value:
                    value = copy.deepcopy(value)
                    path = value["path"]
                    if isinstance(path, str) and path.startswith("depth_anything_3."):
                        value["path"] = f"vggt.models.{path}"
                out[key] = self._rewrite_object_paths(value)
            return out

        if isinstance(cfg, list):
            return [self._rewrite_object_paths(item) for item in cfg]

        return cfg

    def _maybe_normalize_images(self, images: torch.Tensor, normalize_images: bool = True) -> torch.Tensor:
        if not normalize_images:
            return images

        img_min = images.amin().item()
        img_max = images.amax().item()
        if img_min >= -1e-3 and img_max <= 1.0 + 1e-3:
            images = (images - self._imagenet_mean) / self._imagenet_std
        return images

    @staticmethod
    def _infer_sh_degree(gs_world) -> int:
        harmonics = getattr(gs_world, "harmonics", None)
        if harmonics is None or harmonics.shape[-2] <= 0:
            return 0

        coeff_count = int(harmonics.shape[-1])
        sh_degree = int(coeff_count**0.5) - 1
        return max(sh_degree, 0)

    def _compute_manual_metric_scales(self, depth: torch.Tensor, intrinsics: torch.Tensor):
        if self.manual_metric_divisor <= 0:
            raise ValueError(f"manual_metric_divisor must be > 0, got {self.manual_metric_divisor}")

        focal = 0.5 * (intrinsics[..., 0, 0] + intrinsics[..., 1, 1])
        if depth.ndim == 5:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None, None]
        elif depth.ndim == 4:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None]
        else:
            raise ValueError(f"Unexpected depth ndim: {depth.ndim}")

        scene_scale = (focal / self.manual_metric_divisor).mean(dim=1)
        return depth_scale, scene_scale

    @staticmethod
    def _scale_extrinsics_translation(extrinsics: torch.Tensor, scene_scale: torch.Tensor) -> torch.Tensor:
        scaled_t = extrinsics[..., :3, 3] * scene_scale[:, None, None]
        top = torch.cat([extrinsics[..., :3, :3], scaled_t.unsqueeze(-1)], dim=-1)
        if extrinsics.shape[-2] == 4:
            bottom = extrinsics[..., 3:4, :]
            return torch.cat([top, bottom], dim=-2)
        return top

    @staticmethod
    def _scale_gaussians(gs_world, scene_scale: torch.Tensor):
        if gs_world is None:
            return gs_world

        scale = scene_scale[:, None, None]
        if hasattr(gs_world, "means") and torch.is_tensor(gs_world.means):
            means = gs_world.means * scale
            scales = gs_world.scales * scale if torch.is_tensor(getattr(gs_world, "scales", None)) else gs_world.scales
            return Gaussians(
                means=means,
                scales=scales,
                rotations=gs_world.rotations,
                harmonics=gs_world.harmonics,
                opacities=gs_world.opacities,
                features=getattr(gs_world, "features", None),
            )
        if isinstance(gs_world, dict):
            out = dict(gs_world)
            if torch.is_tensor(gs_world.get("means", None)):
                out["means"] = gs_world["means"] * scale
            if torch.is_tensor(gs_world.get("scales", None)):
                out["scales"] = gs_world["scales"] * scale
            return out
        return gs_world

    @staticmethod
    def _scale_backend_voxels(backend_voxels, scene_scale: torch.Tensor):
        """Match compact backend-voxel geometry to manual metric scaling."""
        if backend_voxels is None:
            return None

        scaled = dict(backend_voxels)
        centers = backend_voxels.get("voxel_centers", None)
        if isinstance(centers, (list, tuple)):
            scaled["voxel_centers"] = [
                center * scene_scale[min(batch_idx, scene_scale.numel() - 1)].to(center)
                if torch.is_tensor(center)
                else center
                for batch_idx, center in enumerate(centers)
            ]
        elif torch.is_tensor(centers):
            if centers.ndim >= 3 and centers.shape[0] == scene_scale.numel():
                center_scale = scene_scale.reshape(-1, *([1] * (centers.ndim - 1))).to(centers)
                scaled["voxel_centers"] = centers * center_scale
            else:
                scaled["voxel_centers"] = centers * scene_scale.mean().to(centers)

        voxel_size = backend_voxels.get("voxel_size", None)
        if voxel_size is not None:
            voxel_size = torch.as_tensor(voxel_size, device=scene_scale.device, dtype=scene_scale.dtype)
            scaled["voxel_size"] = voxel_size.reshape(-1)[0] * scene_scale
        return scaled

    @staticmethod
    def _cast_gaussians(gs_world, dtype: torch.dtype):
        """Cast Gaussian attributes at an external operator dtype boundary."""
        if gs_world is None:
            return None
        return Gaussians(
            means=gs_world.means.to(dtype=dtype),
            scales=gs_world.scales.to(dtype=dtype),
            rotations=gs_world.rotations.to(dtype=dtype),
            harmonics=gs_world.harmonics.to(dtype=dtype),
            opacities=gs_world.opacities.to(dtype=dtype),
            features=(
                gs_world.features.to(dtype=dtype)
                if torch.is_tensor(getattr(gs_world, "features", None))
                else getattr(gs_world, "features", None)
            ),
        )

    def _build_predictions_from_output(
        self,
        output,
        images: torch.Tensor,
        image_hw: tuple[int, int],
        render_gs: bool,
        sh_degree: int | None,
        timing_callback=None,
    ) -> dict[str, Any]:
        depth = output.depth
        if depth.ndim == 4:
            depth = depth.unsqueeze(-1)

        depth_conf = output.get("depth_conf", None)
        # voxel_depth_conf = output.get("voxel_depth_conf", None)
        gaussian_voxel_depth_conf = output.get("gaussian_voxel_depth_conf", None)
        highres_backend_voxel_point_ratio = output.get("highres_backend_voxel_point_ratio", None)
        backend_voxels = output.get("backend_voxels", None)
        if depth_conf is None:
            depth_conf = torch.ones_like(depth[..., 0])

        extrinsics = output.extrinsics
        intrinsics = output.intrinsics

        if self.use_manual_metric_scaling:
            depth_scale, scene_scale = self._compute_manual_metric_scales(depth, intrinsics)
            depth = depth * depth_scale
            extrinsics = self._scale_extrinsics_translation(extrinsics, scene_scale)
            backend_voxels = self._scale_backend_voxels(backend_voxels, scene_scale)

        if extrinsics.shape[-2:] == (4, 4):
            extrinsics = extrinsics[..., :3, :]

        pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_hw)

        predictions = {
            "pose_enc": pose_enc,
            "pose_enc_list": [pose_enc],
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth": depth,
            "depth_conf": depth_conf,
            # "voxel_depth_conf": voxel_depth_conf,
            "gaussian_voxel_depth_conf": gaussian_voxel_depth_conf,
        }

        if highres_backend_voxel_point_ratio is not None:
            predictions["highres_backend_voxel_point_ratio"] = highres_backend_voxel_point_ratio
        if backend_voxels is not None:
            predictions["backend_voxels"] = backend_voxels

        if "gaussians" not in output:
            return predictions

        gs_world = output.gaussians
        if self.use_manual_metric_scaling:
            gs_world = self._scale_gaussians(gs_world, scene_scale)
        gs_world = self._normalize_gaussians_opacity_shape(gs_world)

        # The CUDA Gaussian rasterizers used below accept FP32 inputs only.
        # Keep the network and voxel decoder in BF16/FP16, then cast just the
        # final compact Gaussian representation at the renderer boundary.
        gaussian_tensors = (
            gs_world.means,
            gs_world.scales,
            gs_world.rotations,
            gs_world.harmonics,
            gs_world.opacities,
        )
        if render_gs and any(tensor.dtype != torch.float32 for tensor in gaussian_tensors):
            gs_world = self._cast_gaussians(gs_world, torch.float32)
            output["gaussians"] = gs_world

        predictions["gs_world"] = gs_world

        if not render_gs:
            return predictions

        if timing_callback is not None:
            timing_callback("inference/gaussian_render", True)
        try:
            with torch.cuda.amp.autocast(enabled=False):
                last_row = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
                last_row[..., 0, 3] = 1.0
                extrinsics_h = torch.cat([extrinsics, last_row], dim=-2)

                cam_list_all = build_gs_camera(
                    K=intrinsics,
                    ext=extrinsics_h,
                    height=image_hw[0],
                    width=image_hw[1],
                    data_device=images.device,
                )
                gs_background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=images.device)
                gs_pipe = SimpleNamespace(
                    convert_SHs_python=False,
                    compute_cov3D_python=False,
                    depth_ratio=0.0,
                    kernel_size=0.0,
                    require_depth=True,
                    debug=False,
                )
                if sh_degree is None:
                    sh_degree = self._infer_sh_degree(gs_world)

                render_pkgs = []
                for batch_idx in range(images.shape[0]):
                    render_pkgs_batch = []
                    for view_idx in range(images.shape[1]):
                        render_pkg = render(
                            cam_list_all[batch_idx][view_idx],
                            gs_world,
                            gs_pipe,
                            gs_background,
                            batch_idx=batch_idx,
                            sh_degree=sh_degree,
                            gs_mode=self.gs_mode,
                        )
                        render_pkgs_batch.append(render_pkg)
                    render_pkgs.append(render_pkgs_batch)

                predictions["GS_render_pkgs"] = render_pkgs
        finally:
            if timing_callback is not None:
                timing_callback("inference/gaussian_render", False)

        return predictions

    @staticmethod
    def _normalize_gaussians_opacity_shape(gs_world):
        if gs_world is None:
            return gs_world
        opacities = getattr(gs_world, "opacities", None)
        if not torch.is_tensor(opacities):
            return gs_world

        if opacities.ndim >= 3 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        elif opacities.ndim >= 3:
            opacities = opacities.reshape(opacities.shape[0], opacities.shape[1], -1)[..., 0]

        if hasattr(gs_world, "opacities"):
            gs_world.opacities = opacities
        elif isinstance(gs_world, dict):
            gs_world["opacities"] = opacities
        return gs_world



    def forward(self, images: torch.Tensor, forward_dict: dict = None,**kwargs):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        _, _, _, H, W = images.shape
        ori_images = copy.deepcopy(images)

        normalize_images = kwargs.get("normalize_images", True)
        images = self._maybe_normalize_images(images, normalize_images=normalize_images)

        infer_gs = kwargs.get("infer_gs", self.infer_gs)
        gs_from_backend = kwargs.get("gs_from_backend", self.gs_from_backend)
        ref_view_strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        extrinsics = kwargs.get("extrinsics", None)
        intrinsics = kwargs.get("intrinsics", None)
        render_gs = kwargs.get("render_gs", True)
        sh_degree = kwargs.get("sh_degree", None)
        valid_image_mask = kwargs.get("valid_image_mask", None)
        if valid_image_mask is not None:
            expected_mask_shape = images.shape[:2] + images.shape[-2:]
            if tuple(valid_image_mask.shape) != tuple(expected_mask_shape):
                raise ValueError(
                    "valid_image_mask must have shape "
                    f"{tuple(expected_mask_shape)}, got {tuple(valid_image_mask.shape)}"
                )
            valid_image_mask = valid_image_mask.to(device=images.device, dtype=torch.bool)
        image_features = None

        timing_callback = kwargs.get("timing_callback", None)
        output = self.model(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            rgb_images=ori_images,
            image_features=image_features,
            infer_gs=infer_gs,
            gs_from_backend=gs_from_backend,
            ref_view_strategy=ref_view_strategy,
            return_backend_voxels=kwargs.get("return_backend_voxels", False),
            valid_image_mask=valid_image_mask,
            timing_callback=timing_callback,
        )
        predictions = self._build_predictions_from_output(
            output,
            images=images,
            image_hw=(H, W),
            render_gs=render_gs,
            sh_degree=sh_degree,
            timing_callback=timing_callback,
        )
        if image_features is not None:
            predictions["image_features"] = image_features

        raw_stage_outputs = output.get("stage_outputs", None)
        if raw_stage_outputs:
            stage_predictions = []
            for stage_idx, stage_output in enumerate(raw_stage_outputs):
                stage_predictions.append(
                    self._build_predictions_from_output(
                        stage_output,
                        images=images,
                        image_hw=(H, W),
                        render_gs=render_gs and stage_idx == len(raw_stage_outputs) - 1,
                        sh_degree=sh_degree,
                        timing_callback=timing_callback,
                    )
                )
            predictions["baseline"] = stage_predictions[0]
            predictions["refined"] = stage_predictions[-1]
            predictions["stages"] = stage_predictions

        return predictions
