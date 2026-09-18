
import torch
import torch.nn as nn

from torch_scatter import scatter_mean, scatter_sum
from .tools.voxel_utils import get_vox_indices, get_vox_centers_from_points


class BackEnd(nn.Module):
    def __init__(self, hash_base=1024, in_dim=1024, out_dim=256,
                 k_neighbors=16, depth=48, voxel_chunk_size=None):
        super(BackEnd, self).__init__()
        self.voxel_chunk_size = (
            None if voxel_chunk_size is None else int(voxel_chunk_size)
        )
        if self.voxel_chunk_size is not None and self.voxel_chunk_size <= 0:
            raise ValueError("voxel_chunk_size must be positive or None")

    @torch.no_grad()
    def hash_fn(self, coords):
        '''
        A simple hash function for voxel coordinates
        '''
        b, x, y, z = coords.unbind(dim=1)
        return ((b.long() << 48)
               | (x.long() << 32)
               | (y.long() << 16)
               |  z.long())

    @staticmethod
    def _scalar_confidence(depth_conf):
        if depth_conf is None:
            return None
        confidence = depth_conf.reshape(depth_conf.shape[0], -1)
        if confidence.shape[1] != 1:
            confidence = confidence.mean(dim=-1, keepdim=True)
        return confidence

    @staticmethod
    def _scatter_min(values, inverse_id, num_voxels):
        output = torch.full(
            (num_voxels, values.shape[-1]),
            torch.inf,
            device=values.device,
            dtype=values.dtype,
        )
        return torch.scatter_reduce(
            output,
            0,
            inverse_id[:, None].expand_as(values),
            values,
            reduce="amin",
            include_self=True,
        )
    
    
    def mean_by_voxel(
        self,
        points,
        feats,
        batch_ids,
        voxel_size,
        bounding_boxes,
        colors=None,
        image_features=None, depth_conf=None,
        return_aggregation_stats=False,
    ):
        '''Compute mean features for each voxel.
        
        Params:
            - points: (N, 3) tensor of point coordinates
            - feats: (N, C) tensor of point features
            - batch_ids: (N,) tensor of batch indices for each point
            - voxel_size: scalar or (3,) tensor defining the size of each voxel
            - bounding_boxes: (B, 2, 3) tensor of min and max coordinates for each batch
            - colors: (N, 3) tensor of point colors (optional)
            - image_features: (N, F) tensor of dense image descriptors (optional)
            - depth_conf: (N, C) tensor of point depth confidence (optional)
        
        Returns:
            - voxel_feats: (M, C) tensor of mean features for each voxel
            - info: dict containing 'unique_indices' which are the voxel indices corresponding to the mean features
        
        '''
        ori_device = feats.device 
        # if feats.shape[0]>1000000000:
        #     device = torch.device('cpu')
        # else:
        device = ori_device
        points = points.to(device)
        feats = feats.to(device)
        batch_ids = batch_ids.to(device)
        bounding_boxes = bounding_boxes.to(device)
        colors = colors.to(device) if colors is not None else None
        image_features = image_features.to(device) if image_features is not None else None
        depth_conf = depth_conf.to(device) if depth_conf is not None else None
        voxel_indices = get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True)
        voxel_hash = self.hash_fn(voxel_indices) 
        unique_hash, inverse_id = torch.unique(voxel_hash, return_inverse=True) # inverse_id表示这个点属于哪个体素
        
        scalar_confidence = self._scalar_confidence(depth_conf)
        if scalar_confidence is None:
            point_weights = torch.ones(
                (points.shape[0], 1), device=device, dtype=feats.dtype
            )
        else:
            point_weights = scalar_confidence.to(dtype=feats.dtype).clamp_min(0)

        num_voxels = unique_hash.shape[0]
        point_counts = torch.bincount(
            inverse_id, minlength=num_voxels
        ).to(dtype=feats.dtype).unsqueeze(-1)
        weight_sums = scatter_sum(point_weights, inverse_id, dim=0)

        def confidence_weighted_mean(values):
            if values is None:
                return None
            weights = point_weights.to(dtype=values.dtype)
            weighted_sum = scatter_sum(values * weights, inverse_id, dim=0)
            weighted_mean = weighted_sum / weight_sums.to(values).clamp_min(1e-12)
            uniform_mean = scatter_mean(values, inverse_id, dim=0)
            return torch.where(weight_sums.to(values) > 0, weighted_mean, uniform_mean)

        voxel_feats = confidence_weighted_mean(feats).to(ori_device)
        voxel_colors = confidence_weighted_mean(colors)
        voxel_colors = voxel_colors.to(ori_device) if voxel_colors is not None else None
        voxel_image_features = confidence_weighted_mean(image_features)
        if voxel_image_features is not None:
            voxel_image_features = voxel_image_features.to(ori_device)
        voxel_depth_conf = (
            self._scatter_min(scalar_confidence, inverse_id, num_voxels).to(ori_device)
            if scalar_confidence is not None
            else None
        )
        voxel_centers = get_vox_centers_from_points(
            points,
            depth_conf=depth_conf,
            inverse_id=inverse_id,
            num_voxels=num_voxels,
        ).to(ori_device)

        aggregation_stats = None
        if return_aggregation_stats:
            center_weights = weight_sums
            aggregation_stats = {
                "feature_weighted_sum": scatter_sum(
                    feats * point_weights.to(feats), inverse_id, dim=0
                ),
                "feature_sum": scatter_sum(feats, inverse_id, dim=0),
                "color_weighted_sum": (
                    scatter_sum(colors * point_weights.to(colors), inverse_id, dim=0)
                    if colors is not None else None
                ),
                "color_sum": scatter_sum(colors, inverse_id, dim=0) if colors is not None else None,
                "image_feature_weighted_sum": (
                    scatter_sum(image_features * point_weights.to(image_features), inverse_id, dim=0)
                    if image_features is not None else None
                ),
                "image_feature_sum": (
                    scatter_sum(image_features, inverse_id, dim=0)
                    if image_features is not None else None
                ),
                "depth_conf_min": voxel_depth_conf,
                "point_counts": point_counts,
                "weighted_center_sum": voxel_centers * center_weights,
                "center_weights": center_weights,
                "point_center_sum": voxel_centers * point_counts,
            }

        original_indices = torch.arange(voxel_hash.shape[0], device=voxel_hash.device)
        min_original_indices_per_unique_id = torch.full((unique_hash.shape[0],),
                                                voxel_hash.shape[0],
                                                dtype=torch.long,
                                                device=device)
        
        first_occurrence_original_indices = torch.scatter_reduce(
            min_original_indices_per_unique_id,
            0,
            inverse_id,
            original_indices,
            reduce="amin",
            include_self=False
        )

        unique_voxel_indices = voxel_indices[first_occurrence_original_indices].to(ori_device)
        

        # Release intermediate tensors to free VRAM (inference only)
        if not self.training:
            del voxel_hash, unique_hash, inverse_id, original_indices
            del min_original_indices_per_unique_id, first_occurrence_original_indices

        info = {
            'unique_indices': unique_voxel_indices,
            'voxel_centers': voxel_centers,
            'colors': voxel_colors,
            'image_features': voxel_image_features,
            'depth_conf': voxel_depth_conf,
        }    
        if aggregation_stats is not None:
            info["aggregation_stats"] = aggregation_stats

        return voxel_feats, info

    def mean_by_voxel_in_chunks(
        self,
        points,
        feats,
        voxel_size,
        bounding_boxes,
        colors=None,
        image_features=None,
        depth_conf=None,
    ):
        """Aggregate high-resolution points in chunks using one global voxel grid."""
        global_indices = None
        global_stats = None

        def merge_aggregation_stats(
            existing_indices,
            existing_stats,
            local_indices,
            local_stats,
        ):
            if existing_indices is None:
                return local_indices, local_stats

            all_indices = torch.cat([existing_indices, local_indices], dim=0)
            all_hashes = self.hash_fn(all_indices)
            unique_hash, inverse_id = torch.unique(all_hashes, return_inverse=True)

            original_indices = torch.arange(
                all_hashes.shape[0], device=all_hashes.device
            )
            first_indices = torch.full(
                (unique_hash.shape[0],),
                all_hashes.shape[0],
                dtype=torch.long,
                device=all_hashes.device,
            )
            first_indices = torch.scatter_reduce(
                first_indices,
                0,
                inverse_id,
                original_indices,
                reduce="amin",
                include_self=False,
            )
            merged_stats = {}
            for name, local_value in local_stats.items():
                existing_value = existing_stats[name]
                if existing_value is None or local_value is None:
                    if existing_value is not None or local_value is not None:
                        raise ValueError(
                            f"Inconsistent optional aggregation stat {name!r} "
                            "across voxel chunks"
                        )
                    merged_stats[name] = None
                    continue
                combined = torch.cat([existing_value, local_value], dim=0)
                if name == "depth_conf_min":
                    merged_stats[name] = self._scatter_min(
                        combined, inverse_id, unique_hash.shape[0]
                    )
                else:
                    merged_stats[name] = scatter_sum(combined, inverse_id, dim=0)
            return all_indices[first_indices], merged_stats

        for start in range(0, points.shape[0], self.voxel_chunk_size):
            end = min(start + self.voxel_chunk_size, points.shape[0])
            local_voxel_feats, info = self.mean_by_voxel(
                points[start:end],
                feats[start:end],
                torch.zeros(end - start, device=points.device, dtype=torch.long),
                voxel_size,
                bounding_boxes,
                colors=colors[start:end] if colors is not None else None,
                image_features=(
                    image_features[start:end]
                    if image_features is not None
                    else None
                ),
                depth_conf=depth_conf[start:end] if depth_conf is not None else None,
                return_aggregation_stats=True,
            )
            global_indices, global_stats = merge_aggregation_stats(
                global_indices,
                global_stats,
                info["unique_indices"],
                info["aggregation_stats"],
            )
            del local_voxel_feats, info

        point_counts = global_stats["point_counts"]
        center_weights = global_stats["center_weights"]

        def merged_weighted_mean(weighted_sum_name, sum_name):
            weighted_sum = global_stats[weighted_sum_name]
            if weighted_sum is None:
                return None
            weighted_mean = weighted_sum / center_weights.to(weighted_sum).clamp_min(1e-12)
            uniform_mean = global_stats[sum_name] / point_counts.to(weighted_sum).clamp_min(1)
            return torch.where(center_weights.to(weighted_sum) > 0, weighted_mean, uniform_mean)

        voxel_feats = merged_weighted_mean("feature_weighted_sum", "feature_sum")
        weighted_center_sum = global_stats["weighted_center_sum"]
        point_center_sum = global_stats["point_center_sum"]
        voxel_centers = torch.where(
            center_weights > 0,
            weighted_center_sum / center_weights.clamp_min(1e-12),
            point_center_sum / point_counts.clamp_min(1),
        )

        voxel_colors = None
        if colors is not None:
            voxel_colors = merged_weighted_mean("color_weighted_sum", "color_sum")
        voxel_image_features = None
        if image_features is not None:
            voxel_image_features = merged_weighted_mean(
                "image_feature_weighted_sum", "image_feature_sum"
            )
        voxel_depth_conf = None
        if depth_conf is not None:
            voxel_depth_conf = global_stats["depth_conf_min"]

        return voxel_feats, {
            "unique_indices": global_indices,
            "voxel_centers": voxel_centers,
            "colors": voxel_colors,
            "image_features": voxel_image_features,
            "depth_conf": voxel_depth_conf,
        }

    
    def forward(self, pts, feats, voxel_sizes, chunk_size=50000, return_voxel_details=False, colors=None, image_features=None, depth_conf=None, skip_interpolation=True, timing_callback=None):
        '''
        Forward pass for the back-end processing.
        
        Params:
            - pts: (Bs, N, 3) tensor of point coordinates
            - feats: (Bs, N, C) tensor of point features
            - voxel_sizes: list of voxel sizes
            - chunk_size: int, number of points to process in each chunk for interpolation
            - colors: (Bs, N, 3) tensor of point colors (optional)
            - image_features: (Bs, N, F) tensor of dense image descriptors (optional)
            - depth_conf: (Bs, N, C) tensor of point depth confidence (optional)
            - skip_interpolation: bool, if True, skips the expensive point interpolation
        '''
        if not skip_interpolation:
            raise ValueError("VoxelTTO removed the coarse backend feedback/interpolation path")

        if isinstance(feats, list):
            if not skip_interpolation:
                raise ValueError("List backend inputs are only supported when skip_interpolation=True")
            Bs = len(feats)
            C = feats[0].shape[-1] if Bs > 0 else 0
        else:
            Bs, C = feats.shape[0], feats.shape[-1]
        # Voxel hashing and sparse conv are both precision-sensitive. Always run the
        # backend core in fp32 regardless of outer autocast state.


        if not isinstance(feats, list) and len(feats.shape) != 3:
            feats = feats.reshape(Bs, -1, C)
            pts = pts.reshape(Bs, -1, 3)
            if colors is not None:
                colors = colors.reshape(Bs, -1, colors.shape[-1])
            if image_features is not None:
                image_features = image_features.reshape(Bs, -1, image_features.shape[-1])
            if depth_conf is not None:
                depth_conf = depth_conf.reshape(Bs, -1, depth_conf.shape[-1])

        # autocast_ctx = torch.amp.autocast("cuda", enabled=False) if pts.is_cuda else contextlib.nullcontext()
        # with autocast_ctx:
        level_feats = []
        voxel_details = []

        for i, voxel_size in enumerate(voxel_sizes):
            interpolated_feats_per_batch = []
            voxel_feat_per_batch = []
            voxel_center_per_batch = []
            voxel_grid_coord_per_batch = []
            voxel_batch_ids_per_batch = []
            voxel_color_per_batch = []
            voxel_image_feature_per_batch = []
            voxel_depth_conf_per_batch = []

            for batch_idx in range(Bs):
                pts_b = pts[batch_idx]      # (N, 3)
                feats_b = feats[batch_idx]  # (N, C)
                colors_b = colors[batch_idx] if colors is not None else None
                image_features_b = image_features[batch_idx] if image_features is not None else None
                depth_conf_b = depth_conf[batch_idx] if depth_conf is not None else None

                if pts_b.numel() == 0:
                    interpolated_feats_per_batch.append(None)
                    if return_voxel_details:
                        voxel_feat_per_batch.append(feats_b.new_zeros((0, feats_b.shape[-1])))
                        voxel_center_per_batch.append(pts_b.new_zeros((0, 3)))
                        voxel_grid_coord_per_batch.append(
                            torch.zeros((0, 3), device=pts_b.device, dtype=torch.long)
                        )
                        voxel_batch_ids_per_batch.append(torch.zeros((0,), device=pts_b.device, dtype=torch.long))
                        voxel_color_per_batch.append(colors_b.new_zeros((0, colors_b.shape[-1])) if colors_b is not None else None)
                        voxel_image_feature_per_batch.append(
                            image_features_b.new_zeros((0, image_features_b.shape[-1]))
                            if image_features_b is not None else None
                        )
                        voxel_depth_conf_per_batch.append(depth_conf_b.new_zeros((0, depth_conf_b.shape[-1])) if depth_conf_b is not None else None)
                    continue

                bounding_boxes_b = torch.zeros((1, 2, 3), device=pts_b.device)
                bounding_boxes_b[:, 0, :] = pts_b.min(dim=0, keepdim=True).values
                bounding_boxes_b[:, 1, :] = pts_b.max(dim=0, keepdim=True).values

                batch_ids_b = torch.zeros(pts_b.shape[0], device=pts_b.device, dtype=torch.long)

                if (
                    skip_interpolation
                    and self.voxel_chunk_size is not None
                    and pts_b.shape[0] > self.voxel_chunk_size
                ):
                    feat_b, info_b = self.mean_by_voxel_in_chunks(
                        pts_b,
                        feats_b,
                        voxel_size,
                        bounding_boxes_b,
                        colors=colors_b,
                        image_features=image_features_b,
                        depth_conf=depth_conf_b,
                    )
                else:
                    feat_b, info_b = self.mean_by_voxel(
                        pts_b,
                        feats_b,
                        batch_ids_b,
                        voxel_size,
                        bounding_boxes_b,
                        colors=colors_b,
                        image_features=image_features_b,
                    depth_conf=depth_conf_b,
                    )
                vox_id_b = info_b['unique_indices']
                coord_b = info_b['voxel_centers']

                interpolated_feats_per_batch.append(None)
                out_feat = feat_b
                out_coord = coord_b
                out_grid_coord = vox_id_b[:, 1:]

                if return_voxel_details:
                    voxel_feat_per_batch.append(out_feat)
                    voxel_center_per_batch.append(out_coord)
                    voxel_grid_coord_per_batch.append(out_grid_coord)
                    voxel_batch_ids_per_batch.append(
                        torch.full(
                            (out_feat.shape[0],),
                            batch_idx,
                            device=out_feat.device,
                            dtype=torch.long,
                        )
                    )
                    voxel_color_per_batch.append(info_b.get('colors', None))
                    voxel_image_feature_per_batch.append(info_b.get('image_features', None))
                    voxel_depth_conf_per_batch.append(info_b.get('depth_conf', None))

            level_feats.append(None)

            if return_voxel_details:
                voxel_details.append(
                    {
                        "voxel_feat": voxel_feat_per_batch,
                        "voxel_centers": voxel_center_per_batch,
                        "voxel_batch_ids": voxel_batch_ids_per_batch,
                        "voxel_grid_coords": voxel_grid_coord_per_batch,
                        "voxel_colors": voxel_color_per_batch,
                        "voxel_image_features": voxel_image_feature_per_batch,
                        "voxel_depth_conf": voxel_depth_conf_per_batch,
                        "voxel_size": voxel_size,
                    }
                )
        
        if return_voxel_details:
            return level_feats, voxel_details
        return level_feats
