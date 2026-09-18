# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
try:
    if int(os.environ.get("OMP_NUM_THREADS", "1")) <= 0:
        raise ValueError
except (TypeError, ValueError):
    os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import glob
import copy
import torch
from vggt.utils.gsply_helpers import save_gaussian_ply
from PIL import Image

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import pycolmap
from safetensors.torch import load_file
from types import SimpleNamespace
from scripts.eval_demo import Eval


from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import affine_inverse, as_homogeneous, unproject_depth_map_to_point_map
from vggt.models.depth_anything_3.model.utils.transform import mat_to_quat, quat_to_mat
from vggt.models.depth_anything_3.utils.pose_align import align_poses_umeyama, transform_points_sim3
from vggt.models.depth_anything_3.utils.sh_helpers import rotate_sh
from vggt.utils.specs import Gaussians
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
from vggt.models.GS.utils.build_camera import build_gs_camera
from vggt.models.GS.gaussian_renderer import render


torch._dynamo.config.accumulated_cache_size_limit = 512

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, default=None, help="Directory containing the scene images")
    parser.add_argument(
        "--sparse_subdir", type=str, default="0", help="Subdirectory name to save outputs under scene_dir/sparse"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Retained because the existing launch.json passes it; feed-forward export
    # always writes one PINHOLE camera per image.
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument("--config_file", type=str, default=None, help="Path to the model config file")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint file")
    parser.add_argument("--input_long_side", type=int, default=518, help="Network input long side")
    parser.add_argument("--input_patch_size", type=int, default=14, help="Network input size multiple")
    parser.add_argument(
        "--use_training_resolution_crop",
        action="store_true",
        default=False,
        help="Use img_size/aspects from config and center-crop exactly like training",
    )
    parser.add_argument(
        "--training_input_height",
        type=int,
        default=None,
        help="Training-crop height override; must be used with --training_input_width",
    )
    parser.add_argument(
        "--training_input_width",
        type=int,
        default=None,
        help="Training-crop width override; must be used with --training_input_height",
    )
    parser.add_argument(
        "--use_gt_camera",
        action="store_true",
        default=False,
        help="Feed GT COLMAP cameras/images from scene_dir/sparse/<gt_sparse_subdir> into models that support it",
    )
    parser.add_argument(
        "--align_gt_camera",
        action="store_true",
        default=False,
        help="Align predicted outputs to GT COLMAP poses without feeding GT cameras into the model",
    )
    parser.add_argument(
        "--gt_sparse_subdir",
        type=str,
        default="gt",
        help="GT COLMAP sparse subdir used by --use_gt_camera, --align_gt_camera, or as the TCO camera prior",
    )
    parser.add_argument(
        "--posescale",
        "--pose_scale",
        dest="pose_scale",
        type=float,
        default=1.0,
        help="Scale applied to GT camera translations as they are read, e.g. 0.001 for millimeters to meters",
    )
    # parser.add_argument(
    #     "--input_img_path", type=str
    # )
    parser.add_argument("--dist_threshold", type=float, default=0.01, help="Distance threshold for overlapping Gaussians")
    parser.add_argument("--overlap_threshold", type=float, default=0.7, help="Overlap threshold for overlapping Gaussians")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 precision")
    parser.add_argument(
        "--voxel_mesh_max_voxels",
        type=int,
        default=200000,
        help=(
            "Maximum voxels converted to cube mesh for CloudCompare; "
            "use 0 to mesh every voxel (can require very large RAM/disk)"
        ),
    )
    parser.add_argument(
        "--no_save_backend_voxels",
        action="store_false",
        dest="save_backend_voxels",
        help="Do not export backend voxel NPZ, point PLY, or CloudCompare mesh PLY",
    )
    parser.set_defaults(save_backend_voxels=True)
    return parser.parse_args()


def run_VGGT(
    model,
    images,
    dtype,
    resolution=518,
    forward_dict=None,
    gt_extrinsics_for_model=None,
    gt_intrinsics_for_model=None,
    tco_extrinsics_for_model=None,
    tco_intrinsics_for_model=None,
    valid_image_mask=None,
):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    # images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        # with torch.cuda.amp.autocast(dtype=dtype):
        #     images = images[None]  # add batch dimension
        #     aggregated_tokens_list, ps_idx = model.aggregator(images)

        # # Predict Cameras
        # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # # Predict Depth Maps
        # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        images = images[None]

        if "depthanything3" in str(type(model)).lower():
            model_kwargs = {}
            if gt_extrinsics_for_model is not None:
                model_kwargs["extrinsics"] = gt_extrinsics_for_model
            if gt_intrinsics_for_model is not None:
                model_kwargs["intrinsics"] = gt_intrinsics_for_model
            if tco_extrinsics_for_model is not None:
                model_kwargs["tco_extrinsics"] = tco_extrinsics_for_model
            if tco_intrinsics_for_model is not None:
                model_kwargs["tco_intrinsics"] = tco_intrinsics_for_model
            if valid_image_mask is not None:
                model_kwargs["valid_image_mask"] = valid_image_mask
            if forward_dict is not None:
                model_kwargs["return_backend_voxels"] = forward_dict.get("return_backend_voxels", False)
            predictions = model(images, verbose=True, forward_dict=forward_dict, **model_kwargs)
        else:
            if any(
                value is not None
                for value in (
                    gt_extrinsics_for_model,
                    gt_intrinsics_for_model,
                    tco_extrinsics_for_model,
                    tco_intrinsics_for_model,
                )
            ):
                raise ValueError("Camera conditioning and TCO camera priors require a DepthAnything3 model")
            predictions = model(images)

        if "extrinsics" in predictions and "intrinsics" in predictions:
            extrinsic = predictions["extrinsics"]
            intrinsic = predictions["intrinsics"]
        else:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        depth_map, depth_conf = predictions['depth'], predictions['depth_conf']

        if "tsdf_mapper" in predictions.keys():
            tsdf_mapper = predictions['tsdf_mapper']
        else:
            tsdf_mapper = None
        # depth_conf_mask = (predictions["depth_conf"] > 5.0).squeeze(0)
        # gs_views_interval = max(predictions["depth"].shape[0] // 12, 1)
        # save_gaussian_ply(
        #     gaussians=predictions["gs_world"],
        #     save_path="test.ply",
        #     ctx_depth=predictions["depth"][0],
        #     shift_and_scale=False,
        #     save_sh_dc_only=True,
        #     gs_views_interval=gs_views_interval,
        #     inv_opacity=True,
        #     prune_by_depth_percent=0.9,
        #     prune_border_gs=True,
        #     match_3dgs_mcmc_dev=False,
        #     conf_mask=depth_conf_mask
        # )

    # NumPy has no native bfloat16 dtype. Convert prediction outputs to FP32
    # only at the CPU/export boundary while keeping the model forward in BF16.
    extrinsic = extrinsic.squeeze(0).detach().float().float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).detach().float().float().cpu().numpy()
    depth_map = depth_map.squeeze(0).detach().float().float().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).detach().float().float().cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf, predictions, tsdf_mapper


def _config_to_plain_container(cfg):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg


def get_train_image_hw_from_config(cfg, default_hw=(518, 518)):
    cfg = _config_to_plain_container(cfg)
    if not isinstance(cfg, dict) or "img_size" not in cfg:
        return default_hw

    def _find_aspects(node):
        if isinstance(node, dict):
            if "aspects" in node and node["aspects"] is not None:
                return node["aspects"]
            for value in node.values():
                aspects = _find_aspects(value)
                if aspects is not None:
                    return aspects
        elif isinstance(node, (list, tuple)):
            for value in node:
                aspects = _find_aspects(value)
                if aspects is not None:
                    return aspects
        return None

    width = int(cfg["img_size"])
    aspects = _find_aspects(cfg)
    aspect = max(float(item) for item in aspects) if aspects else 1.0
    height = int(round(width * aspect))
    return height, width


def get_fixed_image_ids_from_config(cfg, num_images):
    cfg = _config_to_plain_container(cfg)
    if not isinstance(cfg, dict):
        return None

    data_cfg = cfg.get("data", {})
    common_cfg = None
    if isinstance(data_cfg, dict):
        for phase in ("val", "train"):
            phase_cfg = data_cfg.get(phase, {})
            if isinstance(phase_cfg, dict) and "common_config" in phase_cfg:
                common_cfg = phase_cfg["common_config"]
                break
    if not isinstance(common_cfg, dict) or not common_cfg.get("fixed_triplet", False):
        return None

    fixed_ids = common_cfg.get("fixed_triplet_ids")
    if fixed_ids is not None:
        ids = np.asarray(fixed_ids, dtype=np.int64)
    else:
        img_nums = common_cfg.get("img_nums", cfg.get("img_nums", [num_images, num_images]))
        img_per_seq = int(img_nums[0] if isinstance(img_nums, (list, tuple)) else img_nums)
        mode = str(common_cfg.get("fixed_triplet_mode", "first"))

        if mode == "first":
            ids = np.arange(img_per_seq, dtype=np.int64)
        elif mode == "uniform":
            if img_per_seq == 1:
                ids = np.array([0], dtype=np.int64)
            else:
                ids = np.linspace(0, num_images - 1, img_per_seq).round().astype(np.int64)
        elif mode == "seeded":
            seed = int(common_cfg.get("fixed_triplet_seed", cfg.get("seed_value", 42)))
            ids = np.random.default_rng(seed).choice(num_images, img_per_seq, replace=True).astype(np.int64)
        else:
            raise ValueError(f"Unsupported fixed_triplet_mode={mode!r}; use first, uniform, or seeded")

    if (ids < 0).any() or (ids >= num_images).any():
        raise ValueError(f"fixed triplet ids out of range for {num_images} demo images: {ids.tolist()}")
    return ids.tolist()


def load_and_preprocess_images_training_crop(image_path_list, target_hw):
    """Isotropically resize and center-crop to the configured training size."""
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    target_h, target_w = target_hw
    images = []
    original_coords = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        resize_scale = max(target_w / width, target_h / height)
        resized_w = int(np.ceil(width * resize_scale))
        resized_h = int(np.ceil(height * resize_scale))
        left = max((resized_w - target_w) // 2, 0)
        top = max((resized_h - target_h) // 2, 0)

        img = img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        img = img.crop((left, top, left + target_w, top + target_h))

        # The resized source footprint in cropped tensor coordinates. Negative
        # origins encode pixels removed by the center crop.
        original_coords.append(
            np.array([-left, -top, resized_w - left, resized_h - top, width, height])
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(img_np).permute(2, 0, 1))

    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    return images, original_coords, (target_h, target_w)


def load_and_preprocess_images_aspect_padded(image_path_list, long_side=518, patch_size=14):
    """Isotropically resize a same-resolution sequence and pad its short side."""
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if long_side <= 0 or patch_size <= 0:
        raise ValueError("long_side and patch_size must be positive")
    if long_side % patch_size != 0:
        raise ValueError(
            f"long_side ({long_side}) must be a multiple of patch_size ({patch_size})"
        )

    with Image.open(image_path_list[0]) as first_img:
        original_w, original_h = first_img.size
    resize_scale = long_side / max(original_w, original_h)
    if original_w >= original_h:
        resized_w = long_side
        resized_h = max(1, int(round(original_h * resize_scale)))
    else:
        resized_h = long_side
        resized_w = max(1, int(round(original_w * resize_scale)))

    target_w = ((resized_w + patch_size - 1) // patch_size) * patch_size
    target_h = ((resized_h + patch_size - 1) // patch_size) * patch_size
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2

    images = []
    original_coords = []

    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        if (width, height) != (original_w, original_h):
            raise ValueError(
                "All input images must have the same resolution; "
                f"expected {(original_w, original_h)}, got {(width, height)} for {image_path}"
            )
        img = img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded_img.paste(img, (left, top))

        # Resized image footprint in padded tensor coordinates, then original size.
        original_coords.append(
            np.array([left, top, left + resized_w, top + resized_h, width, height])
        )

        img_np = np.asarray(padded_img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(img_np).permute(2, 0, 1))

    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    return images, original_coords, (target_h, target_w)



def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _iter_colmap_data_lines(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_colmap_text_cameras(cameras_path):
    cameras = {}
    for line in _iter_colmap_data_lines(cameras_path):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(value) for value in parts[4:]], dtype=np.float64)
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported GT COLMAP camera model {model!r} in {cameras_path}")
        cameras[camera_id] = {
            "width": width,
            "height": height,
            "intrinsic": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        }
    return cameras


def read_colmap_text_images(images_path, pose_scale=1.0):
    if not np.isfinite(pose_scale) or pose_scale <= 0:
        raise ValueError(f"pose_scale must be a positive finite number, got {pose_scale}")
    images = {}
    with open(images_path, "r", encoding="utf-8") as file:
        data_lines = [line.strip() for line in file if not line.lstrip().startswith("#")]

    for line in data_lines[0::2]:
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Invalid COLMAP image line in {images_path}: {line}")
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64) * float(pose_scale)
        camera_id = int(parts[8])
        image_name = " ".join(parts[9:])
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = qvec_to_rotmat(qvec)
        extrinsic[:3, 3] = tvec
        images[image_name] = {"camera_id": camera_id, "extrinsic": extrinsic}
    return images


def scale_intrinsic_to_model_input(intrinsic, original_coord):
    scaled = intrinsic.copy()
    x0, y0, x1, y1, original_w, original_h = original_coord
    resize_scale_x = (x1 - x0) / original_w
    resize_scale_y = (y1 - y0) / original_h
    scaled[0, :] *= resize_scale_x
    scaled[1, :] *= resize_scale_y
    scaled[0, 2] += x0
    scaled[1, 2] += y0
    return scaled


def load_gt_camera_from_colmap_sparse(gt_sparse_dir, image_names, original_coords, device, dtype=torch.float32, pose_scale=1.0):
    cameras_path = os.path.join(gt_sparse_dir, "cameras.txt")
    images_path = os.path.join(gt_sparse_dir, "images.txt")
    if not os.path.isfile(cameras_path):
        raise FileNotFoundError(f"GT cameras.txt not found: {cameras_path}")
    if not os.path.isfile(images_path):
        raise FileNotFoundError(f"GT images.txt not found: {images_path}")

    cameras = read_colmap_text_cameras(cameras_path)
    images = read_colmap_text_images(images_path, pose_scale=pose_scale)
    cameras_by_name = {}
    for image_name, original_coord in zip(image_names, original_coords):
        if image_name not in images:
            raise KeyError(f"Image {image_name!r} not found in {images_path}")
        image_entry = images[image_name]
        camera = cameras[image_entry["camera_id"]]
        if image_name in cameras_by_name:
            raise ValueError(f"Duplicate input image name while loading GT cameras: {image_name!r}")
        cameras_by_name[image_name] = {
            "extrinsic": image_entry["extrinsic"].copy(),
            "intrinsic": scale_intrinsic_to_model_input(camera["intrinsic"], original_coord),
        }

    extrinsics, intrinsics = stack_gt_cameras_by_image_name(
        cameras_by_name,
        image_names,
        device=device,
        dtype=dtype,
    )
    return extrinsics, intrinsics, cameras_by_name


def stack_gt_cameras_by_image_name(cameras_by_name, image_names, device, dtype=torch.float32):
    """Build a camera batch in image tensor order, never COLMAP IMAGE_ID order."""
    missing_names = [image_name for image_name in image_names if image_name not in cameras_by_name]
    if missing_names:
        raise KeyError(f"GT cameras missing for input images: {missing_names}")

    extrinsics = np.stack(
        [cameras_by_name[image_name]["extrinsic"] for image_name in image_names]
    )
    intrinsics = np.stack(
        [cameras_by_name[image_name]["intrinsic"] for image_name in image_names]
    )
    extrinsics = torch.from_numpy(extrinsics).to(device=device, dtype=dtype)
    intrinsics = torch.from_numpy(intrinsics).to(device=device, dtype=dtype)
    return extrinsics.unsqueeze(0), intrinsics.unsqueeze(0)


def normalize_gt_extrinsics_for_da3(extrinsics):
    # Mirrors Depth-Anything-3 API: normalize to the first camera frame and median camera distance.
    extrinsics = as_homogeneous(extrinsics.float())
    normalized = torch.matmul(extrinsics, affine_inverse(extrinsics[:, :1]))
    c2w = affine_inverse(normalized)
    camera_distances = c2w[:, :, :3, 3].norm(dim=-1)
    scene_scale = torch.median(camera_distances).clamp(min=1e-1)
    normalized = normalized.clone()
    normalized[:, :, :3, 3] = normalized[:, :, :3, 3] / scene_scale
    return normalized


def transform_gaussians_sim3(gaussians, scale, rot, trans):
    if gaussians is None:
        return None
    dtype = gaussians.means.dtype
    device = gaussians.means.device
    rot_torch = torch.from_numpy(rot).to(dtype=dtype, device=device).float()
    trans_torch = torch.from_numpy(trans).to(dtype=dtype, device=device).float()
    scale_torch = torch.tensor(scale, dtype=dtype, device=device).float()

    means_flat = gaussians.means.reshape(-1, 3)
    means_new = (scale_torch * torch.matmul(means_flat, rot_torch.t()) + trans_torch).reshape(gaussians.means.shape)
    scales_new = gaussians.scales * scale_torch

    rotations_wxyz = gaussians.rotations
    rotations_xyzw = rotations_wxyz[..., [1, 2, 3, 0]]
    rot_mat = quat_to_mat(rotations_xyzw.reshape(-1, 4)).reshape(rotations_wxyz.shape[:-1] + (3, 3))
    rot_mat = torch.matmul(rot_torch, rot_mat)
    rotations_xyzw = mat_to_quat(rot_mat.reshape(-1, 3, 3)).reshape(rotations_wxyz.shape)
    rotations_new = rotations_xyzw[..., [3, 0, 1, 2]]

    harmonics_shape = gaussians.harmonics.shape
    harmonics_flat = gaussians.harmonics.reshape(-1, harmonics_shape[-2], harmonics_shape[-1])
    harmonics_new = rotate_sh(
        harmonics_flat.unsqueeze(0),
        rot_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    ).reshape(harmonics_shape)

    return Gaussians(
        means=means_new.float(),
        scales=scales_new.float(),
        rotations=rotations_new.float(),
        harmonics=harmonics_new.float(),
        opacities=gaussians.opacities.float(),
        features=(
            gaussians.features.float()
            if torch.is_tensor(getattr(gaussians, "features", None))
            else None
        ),
    )


def transform_backend_voxels_sim3(backend_voxels, scale, rot, trans):
    """Transform voxel centers while retaining construction-grid indices."""
    if backend_voxels is None:
        return None
    transformed = dict(backend_voxels)
    centers = backend_voxels.get("voxel_centers", None)
    if isinstance(centers, (list, tuple)):
        transformed_centers = []
        for center in centers:
            if not torch.is_tensor(center):
                transformed_centers.append(center)
                continue
            rot_tensor = torch.as_tensor(rot, device=center.device, dtype=center.dtype)
            trans_tensor = torch.as_tensor(trans, device=center.device, dtype=center.dtype)
            transformed_centers.append(
                float(scale) * torch.matmul(center, rot_tensor.t()) + trans_tensor
            )
        transformed["voxel_centers"] = transformed_centers

    voxel_size = backend_voxels.get("voxel_size", None)
    if voxel_size is not None:
        transformed["voxel_size"] = torch.as_tensor(voxel_size) * abs(float(scale))
    return transformed


def save_backend_voxels(backend_voxels, output_dir, voxel_mesh_max_voxels=200000):
    """Save exact backend voxels as numeric data and a viewer-friendly PLY."""
    if not backend_voxels:
        print("Backend voxel export requested, but this model returned no backend voxels")
        return None

    centers_list = backend_voxels.get("voxel_centers", None)
    if not isinstance(centers_list, (list, tuple)):
        print("Backend voxel export requested, but no voxel centers were returned")
        return None

    valid_batches = [
        batch_idx
        for batch_idx, centers in enumerate(centers_list)
        if torch.is_tensor(centers) and centers.numel() > 0
    ]
    if not valid_batches:
        print("Backend voxel export requested, but the voxel set is empty")
        return None

    centers_parts = [
        centers_list[idx].detach().float().cpu().numpy().reshape(-1, 3)
        for idx in valid_batches
    ]
    centers = np.concatenate(centers_parts, axis=0)
    arrays = {
        "centers": centers,
        "batch_ids": np.concatenate(
            [
                np.full(part.shape[0], idx, dtype=np.int32)
                for idx, part in zip(valid_batches, centers_parts)
            ],
            axis=0,
        ),
        "voxel_size": torch.as_tensor(
            backend_voxels.get("voxel_size", np.nan)
        ).detach().float().cpu().numpy(),
        "coordinate_frame": np.asarray("final_output_world"),
    }

    def collect_optional(key, width=None):
        values = backend_voxels.get(key, None)
        if not isinstance(values, (list, tuple)):
            return None
        parts = []
        for idx, centers_part in zip(valid_batches, centers_parts):
            value = values[idx] if idx < len(values) else None
            if not torch.is_tensor(value) or value.shape[0] != centers_part.shape[0]:
                return None
            array = value.detach().float().cpu().numpy()
            parts.append(array.reshape(-1, width) if width is not None else array.reshape(-1))
        return np.concatenate(parts, axis=0)

    grid_coords = collect_optional("voxel_grid_coords", width=3)
    colors = collect_optional("voxel_colors", width=3)
    confidence = collect_optional("voxel_depth_conf")
    if grid_coords is not None:
        arrays["grid_coords"] = grid_coords.astype(np.int32, copy=False)
    if colors is not None:
        arrays["colors_rgb"] = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    if confidence is not None:
        arrays["depth_confidence"] = confidence.astype(np.float32, copy=False)

    npz_path = os.path.join(output_dir, "backend_voxels.npz")
    ply_path = os.path.join(output_dir, "backend_voxels.ply")
    np.savez_compressed(npz_path, **arrays)
    ply_colors = arrays.get("colors_rgb")
    if ply_colors is None:
        ply_colors = np.tile(
            np.array([[51, 178, 255]], dtype=np.uint8), (centers.shape[0], 1)
        )
    trimesh.PointCloud(centers, colors=ply_colors).export(ply_path)
    if voxel_mesh_max_voxels < 0:
        raise ValueError("--voxel_mesh_max_voxels must be non-negative")

    mesh_count = centers.shape[0]
    if voxel_mesh_max_voxels > 0:
        mesh_count = min(mesh_count, voxel_mesh_max_voxels)
    if mesh_count < centers.shape[0]:
        rng = np.random.default_rng(0)
        mesh_indices = np.sort(
            rng.choice(centers.shape[0], size=mesh_count, replace=False)
        )
    else:
        mesh_indices = np.arange(centers.shape[0])

    mesh_centers = centers[mesh_indices]
    mesh_colors = ply_colors[mesh_indices]
    voxel_sizes = np.asarray(arrays["voxel_size"], dtype=np.float32).reshape(-1)
    if voxel_sizes.size == 1:
        mesh_sizes = np.full(mesh_count, voxel_sizes[0], dtype=np.float32)
    else:
        mesh_sizes = voxel_sizes[
            np.clip(batch_ids[mesh_indices], 0, voxel_sizes.size - 1)
        ]

    cube_offsets = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    cube_faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    mesh_vertices = (
        mesh_centers[:, None, :]
        + cube_offsets[None, :, :] * (mesh_sizes[:, None, None] * 0.5)
    ).reshape(-1, 3)
    mesh_faces = (
        cube_faces[None, :, :]
        + (np.arange(mesh_count, dtype=np.int64) * 8)[:, None, None]
    ).reshape(-1, 3)
    mesh_vertex_colors = np.repeat(mesh_colors, 8, axis=0)
    mesh_path = os.path.join(output_dir, "backend_voxels_mesh.ply")
    trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_colors=mesh_vertex_colors,
        process=False,
    ).export(mesh_path)
    print(
        f"Saved {centers.shape[0]:,} backend voxels to {npz_path} "
        f"and visualization point cloud to {ply_path}; "
        f"cube mesh ({mesh_count:,} voxels) to {mesh_path}"
    )
    return npz_path, ply_path, mesh_path


def infer_gaussian_sh_degree(gaussians):
    harmonics = getattr(gaussians, "harmonics", None)
    if harmonics is None or harmonics.shape[-1] <= 0:
        return 0
    return max(int(harmonics.shape[-1] ** 0.5) - 1, 0)


def mask_gaussians_by_confidence(gaussians, confidence, conf_thres_value):
    """Return Gaussians whose rejected entries have zero opacity.

    Zeroing opacity is equivalent to removing the entries for rendering, while
    preserving the batched Gaussian tensor shape expected by the rasterizers.
    """
    if gaussians is None or confidence is None:
        return gaussians, None

    batch_size, gaussian_count = gaussians.means.shape[:2]
    confidence = torch.as_tensor(confidence, device=gaussians.means.device)
    if confidence.numel() != batch_size * gaussian_count:
        return gaussians, None

    confidence_mask = confidence.reshape(batch_size, gaussian_count) > conf_thres_value
    opacity_mask = confidence_mask
    while opacity_mask.ndim < gaussians.opacities.ndim:
        opacity_mask = opacity_mask.unsqueeze(-1)

    masked_gaussians = Gaussians(
        means=gaussians.means,
        scales=gaussians.scales,
        rotations=gaussians.rotations,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities.masked_fill(~opacity_mask, 0),
        features=getattr(gaussians, "features", None),
    )
    return masked_gaussians, confidence_mask


def get_confidence_masked_gaussians(predictions, conf_thres_value):
    """Apply the export confidence policy to Gaussians used for rendering."""
    gaussians = predictions.get("gs_world", None)
    if gaussians is None:
        return None, None, None

    # Voxel backends emit one confidence value per output Gaussian. Prefer it
    # over image-space depth confidence, exactly as the PLY export path does.
    confidence_sources = (
        ("gaussian_voxel_depth_conf", predictions.get("gaussian_voxel_depth_conf", None)),
        ("depth_conf", predictions.get("depth_conf", None)),
    )
    for source_name, confidence in confidence_sources:
        masked_gaussians, confidence_mask = mask_gaussians_by_confidence(
            gaussians,
            confidence,
            conf_thres_value,
        )
        if confidence_mask is not None:
            return masked_gaussians, confidence_mask, source_name

    return gaussians, None, None


def rerender_gs_with_current_camera(
    predictions,
    image_hw,
    data_device,
    gs_mode=None,
    sh_degree=None,
):
    gaussians = predictions.get("gs_world", None)
    extrinsics = predictions.get("extrinsics", None)
    intrinsics = predictions.get("intrinsics", None)
    if gaussians is None or extrinsics is None or intrinsics is None:
        return None

    with torch.cuda.amp.autocast(enabled=False):
        extrinsics_h = as_homogeneous(extrinsics.detach().float())
        intrinsics = intrinsics.detach().float()
        cam_list_all = build_gs_camera(
            K=intrinsics,
            ext=extrinsics_h,
            height=image_hw[0],
            width=image_hw[1],
            data_device=data_device,
        )
        gs_background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=data_device)
        gs_pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            depth_ratio=0.0,
            kernel_size=0.0,
            require_depth=True,
            debug=False,
        )
        if sh_degree is None:
            sh_degree = infer_gaussian_sh_degree(gaussians)

        render_pkgs = []
        for batch_idx in range(extrinsics_h.shape[0]):
            render_pkgs_batch = []
            for view_idx in range(extrinsics_h.shape[1]):
                render_pkg = render(
                    cam_list_all[batch_idx][view_idx],
                    gaussians,
                    gs_pipe,
                    gs_background,
                    batch_idx=batch_idx,
                    sh_degree=sh_degree,
                    gs_mode=gs_mode,
                )
                render_pkgs_batch.append(render_pkg)
            render_pkgs.append(render_pkgs_batch)
    return render_pkgs



    










def demo_fn(args):
    assert not (args.use_gt_camera and args.align_gt_camera), (
        "--use_gt_camera and --align_gt_camera cannot be enabled at the same time"
    )
    align_to_gt_camera = args.use_gt_camera or args.align_gt_camera

    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    if args.bf16:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT/DA3 for camera and depth estimation
    pretrained_dict = None
    if args.checkpoint_path:
        if args.checkpoint_path.endswith(".safetensors"):
            pretrained_dict = load_file(args.checkpoint_path)
        else:
            pretrained_dict = torch.load(args.checkpoint_path, map_location='cpu')

    if args.config_file is None:
        raise ValueError("--config_file is required by the inference-only VoxelTTO runner")
    config_root_path, config_name = args.config_file.rsplit("/", 1)
    from hydra import initialize, compose
    from hydra.utils import instantiate
    with initialize(version_base=None, config_path=config_root_path):
        cfg = compose(config_name=config_name)
    model = instantiate(cfg.model, _recursive_=False)

    def _safe_load(model, state_dict):
        removed_prefix = "model.backend."
        removed_count = sum(key.startswith(removed_prefix) for key in state_dict)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(removed_prefix)
        }
        result = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = len(result.missing_keys), len(result.unexpected_keys)
        print(
            f"Loaded state dict after dropping {removed_count} removed coarse-backend key(s): "
            f"{missing} missing key(s), {unexpected} unexpected key(s)"
        )
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint does not match the retained inference model: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return result

    if pretrained_dict is not None:
        if "model" in pretrained_dict.keys():
            _safe_load(model, pretrained_dict["model"])
        else:
            _safe_load(model, pretrained_dict)

        del pretrained_dict
        torch.cuda.empty_cache()
    else:
        print("No --checkpoint_path provided, using model default initialization")

    model.eval()
    model = model.to(device).to(dtype)
    print(f"Model loaded")
    evaler = Eval()

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse", args.sparse_subdir)
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    fixed_image_ids = get_fixed_image_ids_from_config(cfg, len(image_path_list))
    if fixed_image_ids is not None:
        image_path_list = [image_path_list[i] for i in fixed_image_ids]
        print(f"Using fixed image ids from config: {fixed_image_ids}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    if args.use_training_resolution_crop:
        override_values = (args.training_input_height, args.training_input_width)
        if (override_values[0] is None) != (override_values[1] is None):
            raise ValueError(
                "--training_input_height and --training_input_width must be provided together"
            )
        if override_values[0] is None:
            train_image_hw = get_train_image_hw_from_config(cfg)
        else:
            train_image_hw = tuple(int(value) for value in override_values)
        if min(train_image_hw) <= 0:
            raise ValueError(f"Training input dimensions must be positive, got {train_image_hw}")
        if any(value % args.input_patch_size != 0 for value in train_image_hw):
            raise ValueError(f"Training input dimensions must be divisible by patch size: {train_image_hw}")
        images, original_coords, train_image_hw = load_and_preprocess_images_training_crop(
            image_path_list, target_hw=train_image_hw
        )
        preprocess_mode = "training center-crop"
    else:
        images, original_coords, train_image_hw = load_and_preprocess_images_aspect_padded(
            image_path_list,
            long_side=args.input_long_side,
            patch_size=args.input_patch_size,
        )
        preprocess_mode = "aspect-preserving padding"
    model_input_height, model_input_width = train_image_hw
    img_load_resolution = max(train_image_hw)
    print(
        f"Using {preprocess_mode} model image size: "
        f"{model_input_height}x{model_input_width} (HxW), "
        f"config_training_crop={args.use_training_resolution_crop}"
    )
    images = images.to(device).to(dtype)
    original_coords = original_coords.to(device)
    eval_original_coords = (
        None if args.use_training_resolution_crop else original_coords.cpu().numpy()
    )
    print(f"Loaded {len(images)} images from {image_dir} with tensor shape {tuple(images.shape[-2:])}")

    # Run VGGT at the selected padded or training-crop input size.
    
    # gt_pose = read_colmap_gt("sparse_DTU/set_23_24_33/scan24/sparse/0/images.txt")
    last_row = torch.tensor([0, 0, 0, 1]).expand(len(image_path_list), 1, 4)
    # gt_pose44 = torch.cat([torch.tensor(gt_pose), last_row], dim=1)

    # image_W,image_H,focal_x,focal_y,cx,cy = read_colmap_camera("sparse_DTU/set_23_24_33/scan24/sparse/0/cameras.txt")
    # intrinsic_matrix = np.array([
    #     [focal_x, 0, cx],
    #     [0, focal_y, cy],
    #     [0, 0, 1]
    # ])
    # intrinsic_gt = np.tile(intrinsic_matrix, (len(image_path_list), 1, 1))
    # image_size_gt = np.array([int(image_W),int(image_H)])
    # gt_pose_wo = read_colmap_gt("sparse_DTU/wo_pose/scan24/sparse/0/images.txt")
    # gt_pose_wo44 = torch.cat([torch.tensor(gt_pose_wo), last_row], dim=1)
    forward_dict = {}
    forward_dict['conf_thres_value'] = args.conf_thres_value
    forward_dict['dist_threshold'] = args.dist_threshold
    forward_dict['overlap_threshold'] = args.overlap_threshold
    forward_dict['return_backend_voxels'] = args.save_backend_voxels

    gt_extrinsics_original = None
    gt_intrinsics_original = None
    gt_extrinsics_for_model = None
    gt_intrinsics_for_model = None
    tco_extrinsics_for_model = None
    tco_intrinsics_for_model = None
    gt_cameras_by_name = None
    tco_camera_prior_enabled = (
        hasattr(model, "tco_config")
        and int(getattr(model.tco_config, "steps", 0)) > 0
    )
    if align_to_gt_camera or tco_camera_prior_enabled:
        gt_sparse_dir = os.path.join(args.scene_dir, "sparse", args.gt_sparse_subdir)
        gt_extrinsics_original, gt_intrinsics_original, gt_cameras_by_name = load_gt_camera_from_colmap_sparse(
            gt_sparse_dir=gt_sparse_dir,
            image_names=base_image_path_list,
            original_coords=original_coords.cpu().numpy(),
            device=images.device,
            dtype=torch.float32,
            pose_scale=args.pose_scale,
        )
        if args.use_gt_camera or tco_camera_prior_enabled:
            normalized_gt_extrinsics = normalize_gt_extrinsics_for_da3(gt_extrinsics_original)
        if args.use_gt_camera:
            gt_extrinsics_for_model = normalized_gt_extrinsics
            gt_intrinsics_for_model = gt_intrinsics_original
            print(f"Using GT camera parameters from {gt_sparse_dir} with pose_scale={args.pose_scale:g}")
            print("GT camera input normalized for DA3; predictions will be aligned to the original COLMAP frame")
        elif args.align_gt_camera:
            print(
                f"Using GT camera parameters from {gt_sparse_dir} for output alignment only "
                f"with pose_scale={args.pose_scale:g}; DA3 camera conditioning is disabled"
            )
        if tco_camera_prior_enabled:
            tco_extrinsics_for_model = normalized_gt_extrinsics
            tco_intrinsics_for_model = gt_intrinsics_original
            if not args.use_gt_camera:
                print(
                    f"Using GT camera parameters from {gt_sparse_dir} as TCO priors only "
                    f"with pose_scale={args.pose_scale:g}; DA3 camera conditioning is disabled"
                )

    # Build the exact valid-input mask. A center crop fills the whole tensor;
    # with padding, only the resized image footprint is valid.
    valid_image_mask = np.zeros((len(images), model_input_height, model_input_width), dtype=bool)
    for view_idx, original_coord in enumerate(original_coords.cpu().numpy()):
        x0, y0, x1, y1 = [int(round(value)) for value in original_coord[:4]]
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, model_input_width), min(y1, model_input_height)
        valid_image_mask[view_idx, y0:y1, x0:x1] = True
    valid_image_mask_for_model = torch.from_numpy(valid_image_mask).to(device=images.device).unsqueeze(0)

    extrinsic, intrinsic, depth_map, depth_conf, predictions, tsdf_mapper = run_VGGT(
        model,
        images,
        dtype,
        model_input_height,
        forward_dict=forward_dict,
        gt_extrinsics_for_model=gt_extrinsics_for_model,
        gt_intrinsics_for_model=gt_intrinsics_for_model,
        tco_extrinsics_for_model=tco_extrinsics_for_model,
        tco_intrinsics_for_model=tco_intrinsics_for_model,
        valid_image_mask=valid_image_mask_for_model,
    )

    # Never create reconstruction points from the artificial black padding.
    depth_conf = np.where(valid_image_mask, depth_conf, -np.inf)
    if predictions.get("depth_conf") is not None:
        prediction_valid_mask = torch.from_numpy(valid_image_mask).to(
            device=predictions["depth_conf"].device
        )
        while prediction_valid_mask.ndim < predictions["depth_conf"].ndim:
            prediction_valid_mask = prediction_valid_mask.unsqueeze(0)
        predictions["depth_conf"] = predictions["depth_conf"].masked_fill(
            ~prediction_valid_mask, -torch.inf
        )
    
    extrinsic44 = torch.cat([torch.tensor(extrinsic), last_row], dim=1)
    extrinsic44 = torch.inverse(extrinsic44)

    # Build the point cloud once in the untouched prediction frame. If GT cameras
    # are requested, align these 3D points directly with the same Sim(3) used for
    # Gaussian means instead of re-unprojecting scaled depth with GT cameras.
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    pose_alignment = None
    if align_to_gt_camera:
        pred_ext_np = as_homogeneous(extrinsic)
        gt_ext_np = gt_extrinsics_original.detach().float().cpu().numpy()[0]
        pred_to_gt_rot, pred_to_gt_trans, pred_to_gt_scale, aligned_pred_ext_np = align_poses_umeyama(
            gt_ext_np,
            pred_ext_np,
            return_aligned=True,
            ransac=gt_ext_np.shape[0] >= 10,
            random_state=42,
        )
        pred_to_gt_scale = float(pred_to_gt_scale)
        if not np.isfinite(pred_to_gt_scale) or abs(pred_to_gt_scale) < 1e-8:
            raise ValueError(f"Invalid predicted-to-GT camera alignment scale: {pred_to_gt_scale}")
        pose_alignment = (pred_to_gt_scale, pred_to_gt_rot, pred_to_gt_trans)

    # Align and evaluate the untouched predicted poses here. In GT-alignment mode,
    # reuse the exact Sim(3) that will also transform cameras, points, and GS.
    pose_metrics_pred = evaler.evaluate_gt_pose_if_available(
        extrinsic44.cpu().numpy(),
        base_image_path_list,
        args.scene_dir,
        sparse_reconstruction_dir,
        gt_sparse_dir=(
            os.path.join(args.scene_dir, "sparse", args.gt_sparse_subdir)
            if align_to_gt_camera
            else None
        ),
        pose_alignment=pose_alignment,
        gt_pose_scale=args.pose_scale if align_to_gt_camera else 1.0,
    )

    if align_to_gt_camera:
        points_3d = transform_points_sim3(
            points_3d,
            pred_to_gt_rot,
            pred_to_gt_trans,
            pred_to_gt_scale,
        )

        if predictions.get("gs_world", None) is not None:
            predictions["gs_world"] = transform_gaussians_sim3(
                predictions["gs_world"],
                pred_to_gt_scale,
                pred_to_gt_rot,
                pred_to_gt_trans,
            )
            print(f"Aligned GS scene to final GT COLMAP frame with scale={pred_to_gt_scale:.6g}")

        if predictions.get("backend_voxels", None) is not None:
            predictions["backend_voxels"] = transform_backend_voxels_sim3(
                predictions["backend_voxels"],
                pred_to_gt_scale,
                pred_to_gt_rot,
                pred_to_gt_trans,
            )
        depth_map = depth_map * pred_to_gt_scale
        predictions["depth"] = predictions["depth"] * pred_to_gt_scale
        extrinsic = aligned_pred_ext_np[:, :3, :].astype(np.float32, copy=False)
        predictions["extrinsics"] = torch.from_numpy(extrinsic).to(
            device=images.device,
            dtype=predictions["depth"].dtype,
        )[None]
        # Keep the model-predicted intrinsics. Only the predicted camera poses are
        # globally aligned; they are never replaced with GT extrinsics.
        predictions["intrinsics"] = torch.from_numpy(intrinsic).to(
            device=images.device,
            dtype=predictions["depth"].dtype,
        )[None]

        extrinsic44 = torch.inverse(
            torch.cat([torch.from_numpy(extrinsic), last_row], dim=1)
        )
        print(f"Aligned predicted cameras and DA3 geometry to the GT COLMAP frame with scale={pred_to_gt_scale:.6g}")
    
    # flip_x_transform = torch.tensor([
    # [1.0, 0, 0, 0],
    # [0, -1, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]
    # ])
    # extrinsic44_trans = []
    # for extrinsic44_1 in extrinsic44:
    #     extrinsic44_1_trans = flip_x_transform@ extrinsic44_1
    #     extrinsic44_trans.append(extrinsic44_1_trans)
    # extrinsic44 = torch.stack(extrinsic44_trans)
    # gt_pose_wo44[:,:3,3] = gt_pose_wo44[:,:3,3] * 10
    # s, R, T = align_multiple_poses(extrinsic44,gt_pose44)
    
    # extrinsic44_trans_inverse = rotate_cameras_with_srt(extrinsic44,s,R,T)
    

    # visualize_camera_poses(rotate_cameras_with_srt(extrinsic44,s,R,T),gt_pose44)

    
    
    
    tsdf = None
    if tsdf_mapper is not None:
        points3d_flatten_tensor = torch.tensor(points_3d.reshape(-1,3)).cuda().unsqueeze(0)

        batch_size = 10000
        tsdf_chunks = []
        for idx in range(0, points3d_flatten_tensor.shape[1], batch_size):
            batch_points = points3d_flatten_tensor[:, idx:idx+batch_size, :]
            batch_tsdf = tsdf_mapper(batch_points.float())
            tsdf_chunks.append(batch_tsdf)
        tsdf = torch.cat(tsdf_chunks, dim=1)

    
    
    # gt_cloud_path = "/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_23_24_33/scan24/sparse/0/points3D.ply"
    # pcd = o3d.io.read_point_cloud(gt_cloud_path)
    # points_trans = rotate_points_with_srt(torch.tensor(points_3d[0].reshape(-1, 3)).float(),s,R,T).numpy()

    # vis_o3d_pcd_2(np.array(pcd.points),points_trans,color1=[1,0,0],color2=[0,1,0])

    
    conf_thres_value = args.conf_thres_value
    max_points_for_colmap = 100000  # randomly sample 3D points
    shared_camera = False  # in the feedforward manner, we do not support shared camera
    camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

    image_size = np.array([model_input_width, model_input_height])
    num_frames, height, width, _ = points_3d.shape

    # points_rgb = F.interpolate(
    #     images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
    # )
    points_rgb = images
    points_rgb = (points_rgb.cpu().float().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= conf_thres_value
    # at most writing 100000 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = points_3d[conf_mask]

    # points_3d_trans2gt = rotate_points_with_srt(torch.tensor(points_3d).float(),s,R,T).numpy()

    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    # reconstruction_gt = batch_np_matrix_to_pycolmap_wo_track(
    #     points_3d_trans2gt,
    #     points_xyf,
    #     points_rgb,
    #     torch.inverse(gt_pose44)[:, :3, :],
    #     intrinsic_gt,
    #     image_size_gt,
    #     shared_camera=shared_camera,
    #     camera_type=camera_type,
    # )

    reconstruction_resolution = img_load_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
        rescale_camera=True,
    )

    # reconstruction_gt = rename_colmap_recons_and_rescale_camera(
    #     reconstruction_gt,
    #     base_image_path_list,
    #     original_coords.cpu().numpy(),
    #     img_size=max(original_coords[0,-2:].cpu().numpy()),
    #     shift_point2d_to_original_res=True,
    #     shared_camera=shared_camera
    # )

    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    if args.save_backend_voxels:
        save_backend_voxels(
            predictions.get("backend_voxels", None),
            sparse_reconstruction_dir,
            voxel_mesh_max_voxels=args.voxel_mesh_max_voxels,
        )
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points_pcd.ply"))

    evaler.evaluate_model_regressed_depth(
        depth_map=depth_map,
        image_path_list=image_path_list,
        scene_dir=args.scene_dir,
        output_dir=sparse_reconstruction_dir,
        target_hw=train_image_hw,
        original_coords=eval_original_coords,
        pred_c2w=extrinsic44.cpu().numpy(),
        pose_image_names=base_image_path_list,
    )
    
    if "gs_world" in predictions.keys():

        depth_conf_mask = (predictions["depth_conf"] > conf_thres_value).squeeze(0)
        gaussian_voxel_conf_mask = None
        if predictions.get("gaussian_voxel_depth_conf") is not None:
            gaussian_voxel_conf_mask = (predictions["gaussian_voxel_depth_conf"] > conf_thres_value).squeeze(0)
        gs_views_interval = max(predictions["depth"].shape[0] // 12, 1)
        save_gaussian_ply(
            gaussians=predictions["gs_world"],
            save_path=os.path.join(sparse_reconstruction_dir,'points3D.ply'),
            ctx_depth=predictions["depth"][0],
            shift_and_scale=False,
            save_sh_dc_only=False,
            gs_views_interval=gs_views_interval,
            inv_opacity=True,
            prune_by_depth_percent=0.9,
            prune_border_gs=True,
            match_3dgs_mcmc_dev=False,
            conf_mask=depth_conf_mask,
            gaussian_conf_mask=gaussian_voxel_conf_mask,
            
        )

        masked_gaussians, render_conf_mask, render_conf_source = get_confidence_masked_gaussians(
            predictions,
            conf_thres_value,
        )
        if render_conf_mask is not None:
            render_predictions = dict(predictions)
            render_predictions["gs_world"] = masked_gaussians
            render_pkgs = rerender_gs_with_current_camera(
                render_predictions,
                image_hw=images.shape[-2:],
                data_device=images.device,
                gs_mode=getattr(model, "gs_mode", None),
            )
            if render_pkgs is not None:
                predictions["GS_render_pkgs"] = render_pkgs
                kept_gaussians = int(render_conf_mask.sum().item())
                total_gaussians = render_conf_mask.numel()
                print(
                    f"Re-rendered GS after --conf_thres_value={conf_thres_value:g} clipping "
                    f"using {render_conf_source}: kept {kept_gaussians}/{total_gaussians} Gaussians"
                )
        elif "GS_render_pkgs" in predictions:
            print(
                "Gaussian count does not match depth-confidence tensors; retaining the model's "
                "already-filtered GS render"
            )

        if "GS_render_pkgs" in predictions.keys():
            evaler.evaluate_gs_render(
                rendered_imgs=predictions['GS_render_pkgs'][0],
                images=images,
                image_path_list=image_path_list,
                scene_dir=args.scene_dir,
                output_dir=sparse_reconstruction_dir,
                target_hw=train_image_hw,
                original_coords=eval_original_coords,
            )

    # Metrics were aligned and computed from the raw model prediction above;
    # only print the cached result here.
    if pose_metrics_pred:
        print(
            "Pred pose metrics - "
            f"matched: {pose_metrics_pred['num_matched']}, "
            f"TransRMSE: {pose_metrics_pred['translation_rmse']:.6f}, "
            f"TransMean: {pose_metrics_pred['translation_mean']:.6f}, "
            f"RotRMSE: {pose_metrics_pred['rotation_rmse_deg']:.6f} deg, "
            f"RotMean: {pose_metrics_pred['rotation_mean_deg']:.6f} deg"
        )
        print(f"Pred pose metric files saved to {pose_metrics_pred['metric_dir']}")


    # print(f"Saving reconstruction to {args.scene_dir}/sparse/gt")
    # sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/gt")
    # os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    # reconstruction_gt.write(sparse_reconstruction_dir)


    # trimesh.PointCloud(points_3d_trans2gt, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/gt/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False,rescale_camera = True
):
    restored_camera_ids = set()
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        x0, y0, x1, y1, original_w, original_h = original_coords[pyimageid - 1]
        scale_x = (x1 - x0) / original_w
        scale_y = (y1 - y0) / original_h
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(f"Invalid image transform for {pyimage.name}: {original_coords[pyimageid - 1]}")

        if rescale_camera and pyimage.camera_id not in restored_camera_ids:
            pred_params = copy.deepcopy(pycamera.params)
            camera_model = getattr(pycamera.model, "name", None)
            if camera_model is None:
                camera_model = str(pycamera.model).rsplit(".", 1)[-1]
            simple_models = {
                "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
            }
            pinhole_models = {
                "PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                "FOV", "THIN_PRISM_FISHEYE",
            }
            if camera_model in simple_models:
                # A one-focal camera cannot represent the tiny x/y difference
                # caused by integer resize rounding, so use their mean.
                pred_params[0] *= 0.5 * (1.0 / scale_x + 1.0 / scale_y)
                pred_params[1] = (pred_params[1] - x0) / scale_x
                pred_params[2] = (pred_params[2] - y0) / scale_y
            elif camera_model in pinhole_models:
                pred_params[0] /= scale_x
                pred_params[1] /= scale_y
                pred_params[2] = (pred_params[2] - x0) / scale_x
                pred_params[3] = (pred_params[3] - y0) / scale_y
            else:
                raise NotImplementedError(f"Unsupported COLMAP camera model: {pycamera.model}")
            pycamera.params = pred_params
            pycamera.width = int(round(original_w))
            pycamera.height = int(round(original_h))
            restored_camera_ids.add(pyimage.camera_id)

        if shift_point2d_to_original_res:
            for point2D in pyimage.points2D:
                point2D.xy = np.array(
                    [(point2D.xy[0] - x0) / scale_x, (point2D.xy[1] - y0) / scale_y]
                )

    return reconstruction



"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
