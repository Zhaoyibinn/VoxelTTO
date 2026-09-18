import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import roma
import torch
import torch.nn.functional as F


from vggt.models.GS.utils.image_utils import psnr as compute_psnr
from vggt.models.GS.utils.loss_utils import ssim as compute_ssim


class Eval:
    _lpips_model = None

    @staticmethod
    def get_exact_size_resize_crop_params(width, height, target_hw):
        target_h, target_w = target_hw
        resize_scale = max(target_w / width, target_h / height)
        resized_w = int(np.ceil(width * resize_scale))
        resized_h = int(np.ceil(height * resize_scale))
        left = max((resized_w - target_w) // 2, 0)
        top = max((resized_h - target_h) // 2, 0)
        return resize_scale, resized_w, resized_h, left, top

    @staticmethod
    def resize_chw_tensor(image, target_hw, mode="bilinear"):
        if tuple(image.shape[-2:]) == tuple(target_hw):
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=target_hw,
            mode=mode,
            align_corners=False,
        ).squeeze(0)

    @staticmethod
    def get_med_dist_between_poses(poses):
        from scipy.spatial.distance import pdist

        return np.median(pdist([p[:3, 3].numpy() for p in poses]))

    @classmethod
    def align_multiple_poses(cls, src_poses, target_poses):
        n = len(src_poses)
        assert src_poses.shape == target_poses.shape == (n, 4, 4)

        def center_and_z(poses):
            eps = cls.get_med_dist_between_poses(poses) / 10
            return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps * poses[:, :3, 2]))

        R, T, s = roma.rigid_points_registration(
            center_and_z(src_poses),
            center_and_z(target_poses),
            compute_scaling=True,
        )
        return s, R, T

    @staticmethod
    def apply_pose_alignment(poses, s, R, t):
        poses = torch.as_tensor(poses, dtype=torch.float32).clone()
        aligned_poses = poses.clone()
        aligned_poses[:, :3, :3] = R @ poses[:, :3, :3]
        aligned_poses[:, :3, 3] = (s * (R @ poses[:, :3, 3].T)).T + t
        return aligned_poses

    @staticmethod
    def _find_colmap_model_dir(sparse_dir):
        if not os.path.isdir(sparse_dir):
            return None
        for filename in ("images.bin", "images.txt"):
            if os.path.exists(os.path.join(sparse_dir, filename)):
                return sparse_dir
        nested_dir = os.path.join(sparse_dir, "0")
        for filename in ("images.bin", "images.txt"):
            if os.path.exists(os.path.join(nested_dir, filename)):
                return nested_dir
        return sparse_dir

    @staticmethod
    def _qvec_to_rotmat(qvec):
        qvec = np.asarray(qvec, dtype=np.float64)
        norm = np.linalg.norm(qvec)
        if norm == 0:
            return np.eye(3, dtype=np.float64)
        q_w, q_x, q_y, q_z = qvec / norm
        return np.array([
            [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * q_x ** 2 - 2 * q_y ** 2],
        ], dtype=np.float64)

    @staticmethod
    def _w2c_to_c2w(rotation, translation):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = np.asarray(rotation, dtype=np.float64)
        w2c[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return np.linalg.inv(w2c)

    @classmethod
    def _pose_from_pycolmap_image(cls, image):
        cam_from_world = getattr(image, "cam_from_world", None)
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        if cam_from_world is not None:
            for matrix_name in ("matrix", "matrix3x4"):
                matrix_func = getattr(cam_from_world, matrix_name, None)
                if callable(matrix_func):
                    matrix = np.asarray(matrix_func(), dtype=np.float64)
                    if matrix.shape == (3, 4):
                        return cls._w2c_to_c2w(matrix[:3, :3], matrix[:3, 3])
                    if matrix.shape == (4, 4):
                        return np.linalg.inv(matrix)

            rotation = getattr(cam_from_world, "rotation", None)
            translation = getattr(cam_from_world, "translation", None)
            if rotation is not None and translation is not None:
                matrix_func = getattr(rotation, "matrix", None)
                rotation = matrix_func() if callable(matrix_func) else rotation
                return cls._w2c_to_c2w(rotation, translation)

        rotmat_func = getattr(image, "rotmat", None)
        tvec = getattr(image, "tvec", None)
        if callable(rotmat_func) and tvec is not None:
            return cls._w2c_to_c2w(rotmat_func(), tvec)

        qvec = getattr(image, "qvec", None)
        if qvec is not None and tvec is not None:
            return cls._w2c_to_c2w(cls._qvec_to_rotmat(qvec), tvec)

        raise ValueError(f"Cannot read pose from COLMAP image {getattr(image, 'name', '<unknown>')}")

    @classmethod
    def _read_colmap_poses_with_pycolmap(cls, colmap_model_dir):
        reconstruction = pycolmap.Reconstruction(colmap_model_dir)
        poses_by_name = {}
        for image_id in reconstruction.images:
            image = reconstruction.images[image_id]
            poses_by_name[image.name] = cls._pose_from_pycolmap_image(image)
        return poses_by_name

    @classmethod
    def _read_colmap_poses_from_images_txt(cls, images_txt_path):
        poses_by_name = {}
        with open(images_txt_path, "r", encoding="utf-8") as file:
            lines = [raw_line.rstrip("\n") for raw_line in file if not raw_line.lstrip().startswith("#")]

        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue

            parts = line.split()
            if len(parts) < 10:
                raise ValueError(f"Invalid COLMAP image line: {line}")
            qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
            image_name = " ".join(parts[9:])
            poses_by_name[image_name] = cls._w2c_to_c2w(cls._qvec_to_rotmat(qvec), tvec)
            index += 2
        return poses_by_name

    @classmethod
    def read_colmap_poses_by_name(cls, colmap_sparse_dir):
        colmap_model_dir = cls._find_colmap_model_dir(colmap_sparse_dir)
        if colmap_model_dir is None:
            return {}

        try:
            return cls._read_colmap_poses_with_pycolmap(colmap_model_dir)
        except Exception as exc:
            images_txt_path = os.path.join(colmap_model_dir, "images.txt")
            if not os.path.exists(images_txt_path):
                raise RuntimeError(f"Failed to read COLMAP model at {colmap_model_dir}: {exc}") from exc
            print(f"pycolmap failed to read {colmap_model_dir}, falling back to images.txt: {exc}")
            return cls._read_colmap_poses_from_images_txt(images_txt_path)

    @staticmethod
    def _pose_name_keys(name):
        normalized_name = name.replace("\\", "/")
        return [name, normalized_name, os.path.basename(normalized_name)]

    @classmethod
    def _match_gt_poses(cls, gt_poses_by_name, image_names):
        gt_lookup = {}
        for gt_name, pose in gt_poses_by_name.items():
            for key in cls._pose_name_keys(gt_name):
                gt_lookup.setdefault(key, pose)

        matched_indices = []
        matched_gt_poses = []
        matched_names = []
        for index, image_name in enumerate(image_names):
            gt_pose = None
            for key in cls._pose_name_keys(image_name):
                if key in gt_lookup:
                    gt_pose = gt_lookup[key]
                    break
            if gt_pose is None:
                continue
            matched_indices.append(index)
            matched_gt_poses.append(gt_pose)
            matched_names.append(image_name)

        return matched_indices, matched_gt_poses, matched_names

    @staticmethod
    def compute_pose_error_metrics(pred_c2w_aligned, gt_c2w):
        pred_c2w_aligned = np.asarray(pred_c2w_aligned, dtype=np.float64)
        gt_c2w = np.asarray(gt_c2w, dtype=np.float64)

        translation_errors = np.linalg.norm(pred_c2w_aligned[:, :3, 3] - gt_c2w[:, :3, 3], axis=1)
        rot_errors = []
        for pred_pose, gt_pose in zip(pred_c2w_aligned, gt_c2w):
            rot_delta = pred_pose[:3, :3] @ gt_pose[:3, :3].T
            cos_angle = np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
            rot_errors.append(np.degrees(np.arccos(cos_angle)))
        rotation_errors = np.asarray(rot_errors, dtype=np.float64)

        return {
            "num_matched": int(len(translation_errors)),
            "translation_mean": float(np.mean(translation_errors)),
            "translation_median": float(np.median(translation_errors)),
            "translation_rmse": float(np.sqrt(np.mean(translation_errors ** 2))),
            "translation_max": float(np.max(translation_errors)),
            "rotation_mean_deg": float(np.mean(rotation_errors)),
            "rotation_median_deg": float(np.median(rotation_errors)),
            "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_errors ** 2))),
            "rotation_max_deg": float(np.max(rotation_errors)),
            "translation_errors": translation_errors,
            "rotation_errors_deg": rotation_errors,
        }

    @staticmethod
    def _set_axes_equal(ax, points):
        if points.size == 0:
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centers = (mins + maxs) * 0.5
        radius = max(np.max(maxs - mins) * 0.5, 1e-6)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

    @staticmethod
    def _camera_visualization_scale(poses_a, poses_b):
        centers = np.concatenate([poses_a[:, :3, 3], poses_b[:, :3, 3]], axis=0)
        if len(centers) <= 1:
            return 0.1
        trajectory_steps = []
        for centers_i in (poses_a[:, :3, 3], poses_b[:, :3, 3]):
            if len(centers_i) > 1:
                distances = np.linalg.norm(np.diff(centers_i, axis=0), axis=1)
                trajectory_steps.extend(distances[distances > 1e-9].tolist())
        if trajectory_steps:
            return max(float(np.median(trajectory_steps)) * 0.35, 1e-4)
        diagonal = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
        return max(float(diagonal) * 0.04, 1e-4)

    @staticmethod
    def _append_camera_lines(points, lines, colors, pose, scale, frustum_color):
        base_index = len(points)
        camera_points = np.array([
            [0.0, 0.0, 0.0],
            [-0.5, -0.35, 1.0],
            [0.5, -0.35, 1.0],
            [0.5, 0.35, 1.0],
            [-0.5, 0.35, 1.0],
            [0.65, 0.0, 0.0],
            [0.0, 0.65, 0.0],
            [0.0, 0.0, 0.65],
        ], dtype=np.float64) * scale
        camera_points = (pose[:3, :3] @ camera_points.T).T + pose[:3, 3]
        points.extend(camera_points.tolist())

        frustum_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        for start, end in frustum_lines:
            lines.append([base_index + start, base_index + end])
            colors.append(frustum_color)

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.25, 1.0])
        for axis_point, axis_color in zip((5, 6, 7), axis_colors):
            lines.append([base_index, base_index + axis_point])
            colors.append(axis_color)

    @staticmethod
    def _append_trajectory_lines(points, lines, colors, poses, color):
        if len(poses) < 2:
            return
        base_index = len(points)
        centers = poses[:, :3, 3]
        points.extend(centers.tolist())
        for index in range(len(centers) - 1):
            lines.append([base_index + index, base_index + index + 1])
            colors.append(color)

    @classmethod
    def save_open3d_aligned_cameras(cls, pred_c2w_aligned, gt_c2w, save_path):
        pred_c2w_aligned = np.asarray(pred_c2w_aligned, dtype=np.float64)
        gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
        scale = cls._camera_visualization_scale(pred_c2w_aligned, gt_c2w)

        points = []
        lines = []
        colors = []
        gt_color = [0.05, 0.35, 1.0]
        pred_color = [1.0, 0.15, 0.05]
        for pose in gt_c2w:
            cls._append_camera_lines(points, lines, colors, pose, scale, gt_color)
        for pose in pred_c2w_aligned:
            cls._append_camera_lines(points, lines, colors, pose, scale, pred_color)
        cls._append_trajectory_lines(points, lines, colors, gt_c2w, gt_color)
        cls._append_trajectory_lines(points, lines, colors, pred_c2w_aligned, pred_color)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        if not o3d.io.write_line_set(save_path, line_set):
            raise RuntimeError(f"Failed to save Open3D camera visualization to {save_path}")

    @classmethod
    def save_aligned_pose_visualization(cls, pred_c2w_aligned, gt_c2w, save_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred_centers = pred_c2w_aligned[:, :3, 3]
        gt_centers = gt_c2w[:, :3, 3]
        all_centers = np.concatenate([pred_centers, gt_centers], axis=0)
        axis_scale = max(np.linalg.norm(all_centers.max(axis=0) - all_centers.min(axis=0)) * 0.04, 1e-4)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], "o-", color="#1f77b4", label="GT")
        ax.plot(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], "o-", color="#d62728", label="Pred aligned")

        step = max(len(gt_c2w) // 20, 1)
        for pose, color in ((gt_c2w[::step], "#1f77b4"), (pred_c2w_aligned[::step], "#d62728")):
            centers = pose[:, :3, 3]
            z_axes = pose[:, :3, 2]
            ax.quiver(
                centers[:, 0], centers[:, 1], centers[:, 2],
                z_axes[:, 0], z_axes[:, 1], z_axes[:, 2],
                length=axis_scale,
                normalize=True,
                color=color,
                linewidth=0.8,
                alpha=0.65,
            )

        cls._set_axes_equal(ax, all_centers)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("Aligned camera poses")
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)

    def evaluate_gt_pose_if_available(
        self,
        pred_c2w,
        image_names,
        scene_dir,
        output_dir,
        gt_sparse_dir=None,
        pose_alignment=None,
        gt_pose_scale=1.0,
    ):
        if gt_sparse_dir is None:
            gt_sparse_dir = os.path.join(scene_dir, "sparse", "gt")
        if not os.path.isdir(gt_sparse_dir):
            return None

        try:
            gt_poses_by_name = self.read_colmap_poses_by_name(gt_sparse_dir)
        except Exception as exc:
            print(f"Found GT at {gt_sparse_dir}, but failed to read poses: {exc}")
            return None

        matched_indices, matched_gt_poses, matched_names = self._match_gt_poses(gt_poses_by_name, image_names)
        if len(matched_indices) < 2:
            print(
                f"Found GT at {gt_sparse_dir}, but only matched {len(matched_indices)} image(s); "
                "skipping pose error evaluation."
            )
            return None

        pred_c2w_np = np.asarray(pred_c2w, dtype=np.float64)
        pred_matched = torch.from_numpy(pred_c2w_np[matched_indices]).float()
        gt_matched = torch.from_numpy(np.stack(matched_gt_poses, axis=0)).float()
        if not np.isfinite(gt_pose_scale) or gt_pose_scale <= 0:
            raise ValueError(f"gt_pose_scale must be a positive finite number, got {gt_pose_scale}")
        gt_matched[:, :3, 3] *= float(gt_pose_scale)

        if pose_alignment is None:
            s, R, T = self.align_multiple_poses(pred_matched, gt_matched)
        else:
            s, R, T = pose_alignment
            s = torch.as_tensor(s, dtype=pred_matched.dtype)
            R = torch.as_tensor(R, dtype=pred_matched.dtype)
            T = torch.as_tensor(T, dtype=pred_matched.dtype)
        pred_aligned = self.apply_pose_alignment(pred_matched, s, R, T).cpu().numpy()
        gt_matched_np = gt_matched.cpu().numpy()
        metrics = self.compute_pose_error_metrics(pred_aligned, gt_matched_np)

        metric_dir = os.path.join(output_dir, "pred_pose_metric")
        os.makedirs(metric_dir, exist_ok=True)
        metrics_path = os.path.join(metric_dir, "pose_error_metrics.txt")
        vis_path = os.path.join(metric_dir, "aligned_pose_visualization.png")
        camera_vis_path = os.path.join(metric_dir, "aligned_pose_cameras.ply")
        aligned_pose_path = os.path.join(metric_dir, "aligned_poses.npz")

        with open(metrics_path, "w", encoding="utf-8") as file:
            file.write(f"gt_sparse_dir: {gt_sparse_dir}\n")
            file.write(f"num_matched: {metrics['num_matched']}\n")
            file.write(f"sim3_scale: {float(s):.9f}\n")
            file.write(
                f"translation_mean: {metrics['translation_mean']:.9f}\n"
                f"translation_median: {metrics['translation_median']:.9f}\n"
                f"translation_rmse: {metrics['translation_rmse']:.9f}\n"
                f"translation_max: {metrics['translation_max']:.9f}\n"
                f"rotation_mean_deg: {metrics['rotation_mean_deg']:.9f}\n"
                f"rotation_median_deg: {metrics['rotation_median_deg']:.9f}\n"
                f"rotation_rmse_deg: {metrics['rotation_rmse_deg']:.9f}\n"
                f"rotation_max_deg: {metrics['rotation_max_deg']:.9f}\n"
            )
            file.write("\nper_image\ttranslation_error\trotation_error_deg\n")
            for name, trans_error, rot_error in zip(
                matched_names, metrics["translation_errors"], metrics["rotation_errors_deg"]
            ):
                file.write(f"{name}\t{trans_error:.9f}\t{rot_error:.9f}\n")

        np.savez(
            aligned_pose_path,
            image_names=np.asarray(matched_names),
            pred_c2w_aligned=pred_aligned,
            gt_c2w=gt_matched_np,
            translation_errors=metrics["translation_errors"],
            rotation_errors_deg=metrics["rotation_errors_deg"],
            sim3_scale=np.asarray(float(s), dtype=np.float64),
            sim3_rotation=R.cpu().numpy(),
            sim3_translation=T.cpu().numpy(),
        )
        self.save_aligned_pose_visualization(pred_aligned, gt_matched_np, vis_path)
        self.save_open3d_aligned_cameras(pred_aligned, gt_matched_np, camera_vis_path)

        metrics["metric_dir"] = metric_dir
        metrics["metrics_path"] = metrics_path
        metrics["visualization_path"] = vis_path
        metrics["camera_visualization_path"] = camera_vis_path
        metrics["aligned_pose_path"] = aligned_pose_path
        return metrics

    @staticmethod
    def compute_render_metrics(rendered_img, gt_img):
        rendered_img = rendered_img.detach().float().clamp(0.0, 1.0)
        gt_img = gt_img.detach().float().clamp(0.0, 1.0)

        if rendered_img.shape[-2:] != gt_img.shape[-2:]:
            gt_img = F.interpolate(
                gt_img.unsqueeze(0),
                size=rendered_img.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        l1_value = F.l1_loss(rendered_img, gt_img).item()
        psnr_value = compute_psnr(rendered_img.unsqueeze(0), gt_img.unsqueeze(0)).mean().item()
        ssim_value = compute_ssim(rendered_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
        return psnr_value, ssim_value, l1_value

    @classmethod
    def get_lpips_model(cls, device):
        if cls._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "LPIPS is required for GS render evaluation. Install it with `pip install lpips`."
                ) from exc
            cls._lpips_model = lpips.LPIPS(net='vgg').eval()
        return cls._lpips_model.to(device)

    @staticmethod
    def compute_lpips_metric(rendered_img, gt_img, lpips_model):
        rendered_img = rendered_img.detach().float().clamp(0.0, 1.0)
        gt_img = gt_img.detach().float().clamp(0.0, 1.0)

        if rendered_img.shape[-2:] != gt_img.shape[-2:]:
            gt_img = F.interpolate(
                gt_img.unsqueeze(0),
                size=rendered_img.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        rendered_lpips = rendered_img.unsqueeze(0) * 2.0 - 1.0
        gt_lpips = gt_img.unsqueeze(0) * 2.0 - 1.0
        return float(lpips_model(rendered_lpips, gt_lpips).item())

    @staticmethod
    def resize_depth_prediction(pred_depth, target_hw):
        pred_4d = pred_depth.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(pred_4d, size=target_hw, mode='bilinear', align_corners=False)
        return resized.squeeze(0).squeeze(0)

    @staticmethod
    def restore_padded_tensor_to_original(tensor, original_coord, mode='bilinear'):
        """Remove model-input padding and resize a HW/CHW tensor to original pixels."""
        x0, y0, x1, y1, original_w, original_h = [int(round(v)) for v in original_coord]
        if tensor.ndim == 2:
            cropped = tensor[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)
            squeeze_dims = 2
        elif tensor.ndim == 3:
            cropped = tensor[:, y0:y1, x0:x1].unsqueeze(0)
            squeeze_dims = 1
        else:
            raise ValueError(f"Expected HW or CHW tensor, got shape {tuple(tensor.shape)}")
        kwargs = {} if mode == 'nearest' else {'align_corners': False}
        restored = F.interpolate(
            cropped,
            size=(original_h, original_w),
            mode=mode,
            **kwargs,
        )
        return restored.squeeze(0).squeeze(0) if squeeze_dims == 2 else restored.squeeze(0)

    @staticmethod
    def compute_depth_valid_mask(pred_depth, gt_depth, min_depth=1e-3, max_depth=1e6, extra_mask=None):
        valid = torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
        valid &= gt_depth > min_depth
        valid &= gt_depth < max_depth
        valid &= pred_depth > 0
        if extra_mask is not None:
            valid &= extra_mask
        return valid

    @staticmethod
    def least_squares_depth_scale(pred_depth, gt_depth, valid_mask):
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        denom = torch.sum(pred_valid * pred_valid)
        if pred_valid.numel() == 0 or denom <= 0:
            return 1.0
        return float((torch.sum(gt_valid * pred_valid) / denom).item())

    @staticmethod
    def compute_depth_metric_dict(pred_depth, gt_depth, valid_mask):
        pred_valid = pred_depth[valid_mask].clamp(min=1e-6)
        gt_valid = gt_depth[valid_mask].clamp(min=1e-6)

        abs_diff = torch.abs(pred_valid - gt_valid)
        sq_diff = (pred_valid - gt_valid) ** 2
        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        ratio = torch.maximum(gt_valid / pred_valid, pred_valid / gt_valid)

        return {
            'num_valid': int(pred_valid.numel()),
            'abs_rel': float((abs_diff / gt_valid).mean().item()),
            'sq_rel': float((sq_diff / gt_valid).mean().item()),
            'rmse': float(torch.sqrt(sq_diff.mean()).item()),
            'rmse_log': float(torch.sqrt((log_diff ** 2).mean()).item()),
            'mae': float(abs_diff.mean().item()),
            'log10': float(torch.mean(torch.abs(torch.log10(pred_valid) - torch.log10(gt_valid))).item()),
            'delta1': float((ratio < 1.25).float().mean().item()),
            'delta2': float((ratio < 1.25 ** 2).float().mean().item()),
            'delta3': float((ratio < 1.25 ** 3).float().mean().item()),
        }

    @staticmethod
    def average_metric_dict(metric_dicts):
        if not metric_dicts:
            return {}

        keys = [key for key in metric_dicts[0].keys() if key != 'num_valid']
        total_valid = sum(item['num_valid'] for item in metric_dicts)
        averaged = {'num_valid': int(total_valid)}
        if total_valid <= 0:
            for key in keys:
                averaged[key] = float('nan')
            return averaged

        for key in keys:
            weighted_sum = sum(item[key] * item['num_valid'] for item in metric_dicts)
            averaged[key] = float(weighted_sum / total_valid)
        return averaged

    def load_scene_depth_tensor(self, scene_dir, image_path, depth_scale=6553.5, target_hw=None):
        depth_dir = Path(scene_dir) / 'depths'
        if not depth_dir.is_dir():
            return None

        image_stem = Path(image_path).stem
        depth_stem_candidates = [
            image_stem.replace('Image_', 'Depth_', 1),
            image_stem.replace('frame', 'depth', 1),
            image_stem,
        ]
        candidate_paths = []
        for depth_stem in depth_stem_candidates:
            candidate_paths.extend([
                depth_dir / f'{depth_stem}.npy',
                depth_dir / f'{depth_stem}.png',
                depth_dir / f'{depth_stem}.exr',
            ])

        depth_path = next((candidate for candidate in candidate_paths if candidate.is_file()), None)
        if depth_path is None:
            return None

        if depth_path.suffix.lower() == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f'Failed to load depth file: {depth_path}')
            depth = depth.astype(np.float32)

        if depth.ndim != 2:
            raise ValueError(f'Depth file must be HxW: {depth_path}')
        depth[~np.isfinite(depth)] = 0.0
        depth = depth / depth_scale
        depth_tensor = torch.from_numpy(depth)
        if target_hw is not None:
            depth_tensor = self.preprocess_depth_like_rgb_exact_size(depth_tensor, target_hw)
        return depth_tensor

    def preprocess_depth_like_rgb_exact_size(self, depth, target_hw):
        target_h, target_w = target_hw
        depth = depth.detach().float().squeeze()
        height, width = depth.shape

        _, resized_w, resized_h, left, top = self.get_exact_size_resize_crop_params(width, height, target_hw)

        depth_4d = depth.unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth_4d, size=(resized_h, resized_w), mode='nearest').squeeze(0).squeeze(0)

        return depth[top:top + target_h, left:left + target_w]

    def compute_rendered_depth_metrics(self, rendered_depth, gt_depth, render_alpha=None):
        rendered_depth = rendered_depth.detach().float().squeeze()
        gt_depth = gt_depth.detach().float().squeeze()

        if rendered_depth.shape != gt_depth.shape:
            gt_depth = self.resize_depth_prediction(gt_depth, tuple(rendered_depth.shape))

        extra_mask = None
        if render_alpha is not None:
            render_alpha = render_alpha.detach().float().squeeze()
            if render_alpha.shape != rendered_depth.shape:
                render_alpha = self.resize_depth_prediction(render_alpha, tuple(rendered_depth.shape))
            extra_mask = render_alpha > 0

        valid_mask = self.compute_depth_valid_mask(rendered_depth, gt_depth, extra_mask=extra_mask)
        if not valid_mask.any():
            empty_metrics = {'num_valid': 0}
            for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'mae', 'log10', 'delta1', 'delta2', 'delta3']:
                empty_metrics[key] = float('nan')
            return {
                'scale': 1.0,
                'raw': empty_metrics,
                'aligned': dict(empty_metrics),
            }

        raw_metrics = self.compute_depth_metric_dict(rendered_depth, gt_depth, valid_mask)
        scale = self.least_squares_depth_scale(rendered_depth, gt_depth, valid_mask)
        aligned_metrics = self.compute_depth_metric_dict(rendered_depth * scale, gt_depth, valid_mask)
        return {
            'scale': scale,
            'raw': raw_metrics,
            'aligned': aligned_metrics,
        }

    @staticmethod
    def empty_model_depth_metric_dict():
        return {
            'num_valid': 0,
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'mae': float('nan'),
            'log10': float('nan'),
            'delta_1_05': float('nan'),
            'delta_1_03': float('nan'),
            'delta_1_01': float('nan'),
        }

    @staticmethod
    def compute_model_depth_metric_dict(pred_depth, gt_depth, valid_mask):
        pred_valid = pred_depth[valid_mask].clamp(min=1e-6)
        gt_valid = gt_depth[valid_mask].clamp(min=1e-6)

        abs_diff = torch.abs(pred_valid - gt_valid)
        sq_diff = (pred_valid - gt_valid) ** 2
        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        ratio = torch.maximum(gt_valid / pred_valid, pred_valid / gt_valid)

        return {
            'num_valid': int(pred_valid.numel()),
            'abs_rel': float((abs_diff / gt_valid).mean().item()),
            'sq_rel': float((sq_diff / gt_valid).mean().item()),
            'rmse': float(torch.sqrt(sq_diff.mean()).item()),
            'rmse_log': float(torch.sqrt((log_diff ** 2).mean()).item()),
            'mae': float(abs_diff.mean().item()),
            'log10': float(torch.mean(torch.abs(torch.log10(pred_valid) - torch.log10(gt_valid))).item()),
            'delta_1_05': float((ratio < 1.05).float().mean().item()),
            'delta_1_03': float((ratio < 1.03).float().mean().item()),
            'delta_1_01': float((ratio < 1.01).float().mean().item()),
        }

    def compute_pose_scale_for_depth_eval(self, pred_c2w, image_names, scene_dir):
        gt_sparse_dir = os.path.join(scene_dir, "sparse", "gt")
        if pred_c2w is None or not os.path.isdir(gt_sparse_dir):
            return None, 0

        try:
            gt_poses_by_name = self.read_colmap_poses_by_name(gt_sparse_dir)
        except Exception as exc:
            print(f"Found GT at {gt_sparse_dir}, but failed to read poses for depth scale: {exc}")
            return None, 0

        matched_indices, matched_gt_poses, _ = self._match_gt_poses(gt_poses_by_name, image_names)
        if len(matched_indices) < 2:
            print(
                f"Found GT at {gt_sparse_dir}, but only matched {len(matched_indices)} image(s); "
                "model depth pose scale unavailable."
            )
            return None, len(matched_indices)

        pred_c2w_np = np.asarray(pred_c2w, dtype=np.float64)
        pred_matched = torch.from_numpy(pred_c2w_np[matched_indices]).float()
        gt_matched = torch.from_numpy(np.stack(matched_gt_poses, axis=0)).float()
        scale, _, _ = self.align_multiple_poses(pred_matched, gt_matched)
        return float(scale), len(matched_indices)

    def compute_model_regressed_depth_metrics(self, pred_depth, gt_depth):
        pred_depth = pred_depth.detach().float().squeeze()
        gt_depth = gt_depth.detach().float().squeeze()

        if pred_depth.shape != gt_depth.shape:
            gt_depth = self.resize_depth_prediction(gt_depth, tuple(pred_depth.shape))

        valid_mask = self.compute_depth_valid_mask(pred_depth, gt_depth)
        if not valid_mask.any():
            empty_metrics = self.empty_model_depth_metric_dict()
            return {
                'scale': 1.0,
                'raw': empty_metrics,
                'aligned': dict(empty_metrics),
            }

        raw_metrics = self.compute_model_depth_metric_dict(pred_depth, gt_depth, valid_mask)
        scale = self.least_squares_depth_scale(pred_depth, gt_depth, valid_mask)
        aligned_metrics = self.compute_model_depth_metric_dict(pred_depth * scale, gt_depth, valid_mask)

        return {
            'scale': scale,
            'raw': raw_metrics,
            'aligned': aligned_metrics,
        }

    @staticmethod
    def format_model_depth_metrics(prefix, metrics):
        if not metrics or metrics.get('num_valid', 0) <= 0:
            return f'{prefix}_valid: 0'
        return (
            f'{prefix}_valid: {metrics["num_valid"]}'
            f'\t{prefix}_abs_rel: {metrics["abs_rel"]:.6f}'
            f'\t{prefix}_rmse: {metrics["rmse"]:.6f}'
            f'\t{prefix}_delta_1_05: {metrics["delta_1_05"]:.6f}'
            f'\t{prefix}_delta_1_03: {metrics["delta_1_03"]:.6f}'
            f'\t{prefix}_delta_1_01: {metrics["delta_1_01"]:.6f}'
        )

    @staticmethod
    def format_depth_metrics(prefix, metrics):
        if not metrics or metrics.get('num_valid', 0) <= 0:
            return f'{prefix}_valid: 0'
        return (
            f'{prefix}_valid: {metrics["num_valid"]}'
            f'\t{prefix}_abs_rel: {metrics["abs_rel"]:.6f}'
            f'\t{prefix}_rmse: {metrics["rmse"]:.6f}'
            f'\t{prefix}_delta1: {metrics["delta1"]:.6f}'
        )

    @staticmethod
    def depth_to_vis_image(depth_tensor):
        depth_np = depth_tensor.detach().float().cpu().numpy()
        finite_mask = np.isfinite(depth_np) & (depth_np > 0)
        depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
        if finite_mask.any():
            valid_depth = depth_np[finite_mask]
            depth_min = np.percentile(valid_depth, 2.0)
            depth_max = np.percentile(valid_depth, 98.0)
            if depth_max > depth_min:
                normalized = np.clip((depth_np - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                depth_vis = (normalized * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    def evaluate_model_regressed_depth(
        self,
        depth_map,
        image_path_list,
        scene_dir,
        output_dir,
        target_hw,
        original_coords=None,
        pred_c2w=None,
        pose_image_names=None,
    ):
        model_depth_dir = os.path.join(output_dir, 'model_regressed_depth')
        model_depth_save_dir = os.path.join(model_depth_dir, 'depths')
        metrics_path = os.path.join(model_depth_dir, 'metrics.txt')

        os.makedirs(model_depth_dir, exist_ok=True)
        os.makedirs(model_depth_save_dir, exist_ok=True)

        metrics_lines = ["scale_source: depth_least_squares"]
        depth_raw_metrics = []
        depth_aligned_metrics = []

        for i, image_path in enumerate(image_path_list):
            pred_depth = torch.from_numpy(np.asarray(depth_map[i])).float().squeeze()
            if pred_depth.shape != tuple(target_hw):
                pred_depth = self.resize_depth_prediction(pred_depth, target_hw)

            output_hw = tuple(target_hw)
            if original_coords is not None:
                pred_depth = self.restore_padded_tensor_to_original(
                    pred_depth, original_coords[i], mode='bilinear'
                )
                output_hw = tuple(int(round(v)) for v in original_coords[i][-2:][::-1])

            pred_depth_to_save = pred_depth.squeeze()
            np.save(
                os.path.join(model_depth_save_dir, f'model_depth_{i}.npy'),
                pred_depth_to_save.detach().float().cpu().numpy(),
            )
            cv2.imwrite(
                os.path.join(model_depth_save_dir, f'model_depth_{i}.png'),
                self.depth_to_vis_image(pred_depth_to_save),
            )

            gt_depth = self.load_scene_depth_tensor(
                scene_dir,
                image_path,
                depth_scale=6553.5,
                target_hw=output_hw,
            )

            metric_line = f"model_depth_{i}.png"
            if gt_depth is not None:
                depth_metrics = self.compute_model_regressed_depth_metrics(pred_depth, gt_depth)
                depth_raw_metrics.append(depth_metrics['raw'])
                depth_aligned_metrics.append(depth_metrics['aligned'])
                metric_line += (
                    f"	DepthScale: {depth_metrics['scale']:.6f}"
                    f"	{self.format_model_depth_metrics('ModelDepthRaw', depth_metrics['raw'])}"
                    f"	{self.format_model_depth_metrics('ModelDepthAligned', depth_metrics['aligned'])}"
                )
            else:
                metric_line += "	DepthMetrics: unavailable(no matching GT depth)"

            metrics_lines.append(metric_line)

        avg_depth_raw = self.average_metric_dict(depth_raw_metrics)
        avg_depth_aligned = self.average_metric_dict(depth_aligned_metrics)
        if avg_depth_raw:
            summary_line = (
                "average"
                f"	{self.format_model_depth_metrics('ModelDepthRaw', avg_depth_raw)}"
                f"	{self.format_model_depth_metrics('ModelDepthAligned', avg_depth_aligned)}"
            )
            metrics_lines.append(summary_line)

        with open(metrics_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(metrics_lines) + '\n')

        if avg_depth_raw:
            print(
                "Model regressed depth metrics - "
                "scale source: depth least squares, "
                f"raw AbsRel: {avg_depth_raw.get('abs_rel', float('nan')):.6f}, "
                f"raw RMSE: {avg_depth_raw.get('rmse', float('nan')):.6f}, "
                f"raw delta@1.05: {avg_depth_raw.get('delta_1_05', float('nan')):.6f}, "
                f"raw delta@1.03: {avg_depth_raw.get('delta_1_03', float('nan')):.6f}, "
                f"raw delta@1.01: {avg_depth_raw.get('delta_1_01', float('nan')):.6f}, "
                f"aligned AbsRel: {avg_depth_aligned.get('abs_rel', float('nan')):.6f}, "
                f"aligned RMSE: {avg_depth_aligned.get('rmse', float('nan')):.6f}, "
                f"aligned delta@1.05: {avg_depth_aligned.get('delta_1_05', float('nan')):.6f}, "
                f"aligned delta@1.03: {avg_depth_aligned.get('delta_1_03', float('nan')):.6f}, "
                f"aligned delta@1.01: {avg_depth_aligned.get('delta_1_01', float('nan')):.6f}"
            )
        else:
            print("Model regressed depth metrics unavailable: no matching GT depth")
        print(f"Model regressed depth metric files saved to {model_depth_dir}")

    def evaluate_gs_render(
        self,
        rendered_imgs,
        images,
        image_path_list,
        scene_dir,
        output_dir,
        target_hw,
        original_coords=None,
    ):
        rendered_imgs_save_dir = os.path.join(output_dir, 'GS_rendered')
        rendered_imgs_save_dir_rgb = os.path.join(rendered_imgs_save_dir, 'colors')
        rendered_imgs_save_dir_gt = os.path.join(rendered_imgs_save_dir, 'gt')
        rendered_imgs_save_dir_compare = os.path.join(rendered_imgs_save_dir, 'compare')
        rendered_imgs_save_dir_depth = os.path.join(rendered_imgs_save_dir, 'depths')
        rendered_imgs_metrics_path = os.path.join(rendered_imgs_save_dir, 'metrics.txt')

        os.makedirs(rendered_imgs_save_dir, exist_ok=True)
        os.makedirs(rendered_imgs_save_dir_rgb, exist_ok=True)
        os.makedirs(rendered_imgs_save_dir_gt, exist_ok=True)
        os.makedirs(rendered_imgs_save_dir_compare, exist_ok=True)
        os.makedirs(rendered_imgs_save_dir_depth, exist_ok=True)

        metrics_lines = []
        psnr_values = []
        ssim_values = []
        l1_values = []
        lpips_values = []
        depth_raw_metrics = []
        depth_aligned_metrics = []
        lpips_model = self.get_lpips_model(images.device)
        for i in range(len(rendered_imgs)):
            original_coord = original_coords[i] if original_coords is not None else None
            output_hw = tuple(target_hw)
            if original_coord is not None:
                output_hw = (int(round(original_coord[-1])), int(round(original_coord[-2])))
            rendered_img = rendered_imgs[i]['render']
            rendered_depth = rendered_imgs[i].get('depth')
            render_alpha = rendered_imgs[i].get('rend_alpha', rendered_imgs[i].get('alpha'))
            rendered_img = self.resize_chw_tensor(rendered_img, target_hw)
            if rendered_depth is not None:
                rendered_depth = self.resize_depth_prediction(rendered_depth.detach().float().squeeze(), target_hw)
            if render_alpha is not None:
                render_alpha = self.resize_depth_prediction(render_alpha.detach().float().squeeze(), target_hw)
            gt_img = self.resize_chw_tensor(images[i].detach().float().clamp(0.0, 1.0), target_hw)
            if original_coord is not None:
                rendered_img = self.restore_padded_tensor_to_original(rendered_img, original_coord)
                gt_img = self.restore_padded_tensor_to_original(gt_img, original_coord)
                if rendered_depth is not None:
                    rendered_depth = self.restore_padded_tensor_to_original(rendered_depth, original_coord)
                if render_alpha is not None:
                    render_alpha = self.restore_padded_tensor_to_original(render_alpha, original_coord)

            if render_alpha is not None:
                rendered_mask = render_alpha.squeeze() > 0
            else:
                rendered_mask = rendered_img.abs().sum(dim=0) > 0
            rendered_mask = rendered_mask.unsqueeze(0)
            rendered_img = rendered_img.masked_fill(~rendered_mask, 0)
            gt_img = gt_img.masked_fill(~rendered_mask, 0)

            psnr_value, ssim_value, l1_value = self.compute_render_metrics(rendered_img, gt_img)
            lpips_value = self.compute_lpips_metric(rendered_img, gt_img, lpips_model)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            l1_values.append(l1_value)
            lpips_values.append(lpips_value)

            metric_line = f"rendered_{i}.png\tPSNR: {psnr_value:.4f}\tSSIM: {ssim_value:.6f}\tL1: {l1_value:.6f}\tLPIPS: {lpips_value:.6f}"

            gt_depth = self.load_scene_depth_tensor(
                scene_dir,
                image_path_list[i],
                depth_scale=6553.5,
                target_hw=output_hw,
            )
            if rendered_depth is not None:
                rendered_depth_to_save = rendered_depth.squeeze()
                np.save(
                    os.path.join(rendered_imgs_save_dir_depth, f'rendered_{i}.npy'),
                    rendered_depth_to_save.detach().float().cpu().numpy(),
                )
                cv2.imwrite(
                    os.path.join(rendered_imgs_save_dir_depth, f'rendered_{i}.png'),
                    self.depth_to_vis_image(rendered_depth_to_save),
                )

            if gt_depth is not None and rendered_depth is not None:
                rendered_depth = rendered_depth.cuda()
                gt_depth = gt_depth.cuda()
                depth_metrics = self.compute_rendered_depth_metrics(rendered_depth, gt_depth, render_alpha=render_alpha)
                depth_raw_metrics.append(depth_metrics['raw'])
                depth_aligned_metrics.append(depth_metrics['aligned'])
                metric_line += (
                    f"\tDepthScale: {depth_metrics['scale']:.6f}"
                    f"\t{self.format_depth_metrics('DepthRaw', depth_metrics['raw'])}"
                    f"\t{self.format_depth_metrics('DepthAligned', depth_metrics['aligned'])}"
                )
            elif rendered_depth is None:
                metric_line += "\tDepthMetrics: unavailable(render package has no depth)"
            else:
                metric_line += "\tDepthMetrics: unavailable(no matching GT depth)"

            metrics_lines.append(metric_line)

            rendered_img_np = rendered_img.permute(1, 2, 0).detach().float().clamp(0.0, 1.0).cpu().numpy()
            rendered_img_np = cv2.cvtColor((rendered_img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(rendered_imgs_save_dir_rgb, f'rendered_{i}.png'), rendered_img_np)

            gt_img_np = gt_img.permute(1, 2, 0).detach().float().clamp(0.0, 1.0).cpu().numpy()
            gt_img_np = cv2.cvtColor((gt_img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(rendered_imgs_save_dir_gt, f'gt_{i}.png'), gt_img_np)

            diff_img = torch.abs(rendered_img.detach().float().clamp(0.0, 1.0) - gt_img)
            diff_img_np = diff_img.permute(1, 2, 0).cpu().numpy()
            diff_img_np = cv2.cvtColor((np.clip(diff_img_np * 4.0, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            separator = np.full((output_hw[0], 8, 3), 255, dtype=np.uint8)
            compare_img_np = np.concatenate([gt_img_np, separator, rendered_img_np, separator, diff_img_np], axis=1)
            cv2.imwrite(os.path.join(rendered_imgs_save_dir_compare, f'compare_{i}.png'), compare_img_np)

        if psnr_values:
            avg_psnr = float(np.mean(psnr_values))
            avg_ssim = float(np.mean(ssim_values))
            avg_l1 = float(np.mean(l1_values))
            avg_lpips = float(np.mean(lpips_values))
            summary_line = f"average\tPSNR: {avg_psnr:.4f}\tSSIM: {avg_ssim:.6f}\tL1: {avg_l1:.6f}\tLPIPS: {avg_lpips:.6f}"

            avg_depth_raw = self.average_metric_dict(depth_raw_metrics)
            avg_depth_aligned = self.average_metric_dict(depth_aligned_metrics)
            if avg_depth_raw:
                summary_line += (
                    f"\t{self.format_depth_metrics('DepthRaw', avg_depth_raw)}"
                    f"\t{self.format_depth_metrics('DepthAligned', avg_depth_aligned)}"
                )

            metrics_lines.append(summary_line)
            print(
                f"Rendered image metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.6f}, L1: {avg_l1:.6f}, LPIPS: {avg_lpips:.6f}"
            )
            if avg_depth_aligned:
                print(
                    "Rendered depth metrics - "
                    f"aligned AbsRel: {avg_depth_aligned.get('abs_rel', float('nan')):.6f}, "
                    f"RMSE: {avg_depth_aligned.get('rmse', float('nan')):.6f}, "
                    f"delta1: {avg_depth_aligned.get('delta1', float('nan')):.6f}"
                )
            with open(rendered_imgs_metrics_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(metrics_lines) + '\n')
