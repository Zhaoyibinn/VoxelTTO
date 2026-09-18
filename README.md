# VoxelTTO

Inference-only extraction of the configured TCO + high-resolution voxel
Gaussian pipeline.

## Run

```bash
conda activate vggt
OMP_NUM_THREADS=1 python demo_colmap.py \
  --scene_dir data/Replica/gsloc_xiaorong_pro6000/office0_50_500 \
  --shared_camera \
  --conf_thres_value 0.0 \
  --use_training_resolution_crop \
  --align_gt_camera \
  --config_file configs/inference.yaml \
  --checkpoint_path logs/vgvggt_spunet_iggt_224x448/ckpts/checkpoint.pt \
  --sparse_subdir test
```

The image, depth, GT COLMAP and checkpoint paths are symbolic links. Generated
files are written to the local `sparse/test` directory, never into the source
repository.

The configured backend iteration count is zero. Consequently this project keeps
the final high-resolution voxel aggregation and SpUNet Gaussian decoder, while
the low-resolution feedback backend, PointTransformer, KNN interpolation,
training stack and BA tracker are intentionally absent.
