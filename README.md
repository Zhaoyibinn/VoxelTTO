# VoxelTTO
*Voxel-Aligned Feed-Forward 3D Gaussian Splatting with Test-Time Optimization*


## Overview

Recent feed-forward 3D Gaussian Splatting (3DGS) methods typically regress pixel-aligned Gaussian primitives, often causing excessive overlap and artifacts, while inaccuracies in predicted camera poses can lead to misalignment in novel-view synthesis (NVS).
We present VoxelTTO, a feed-forward framework for reconstructing geometrically accurate 3DGS scenes from an arbitrary number of images and optional camera parameters. VoxelTTO aggregates dense image features into a global voxel representation and decodes Gaussians from voxel features, breaking the pixel-to-Gaussian correspondence. To exploit known camera parameters while keeping the pretrained visual foundation model (VFM) parameters frozen, we introduce test-time optimization (TTO) that adapts lightweight LoRA modules using pose supervision. We further replace vanilla 3DGS rasterization with stochastic solid volume rendering during training and inference, improving geometric fidelity. Training updates only the voxel-aligned Gaussian reconstruction modules, requiring 80 GPU hours. Experiments on Replica, Tanks and Temples, and DTU demonstrate improved RGB-D NVS and camera-pose estimation relative to prior methods.

## TODO

- [ ] Plans after publication
  - [ ] Release training code
  - [ ] Release pre-trained weights
  - [ ] Release a minimal demo


## Environment Setup

This code has been tested with Torch 2.7.1 + CUDA 12.8.

First, clone this repository to your local machine, and install the dependencies.

```bash
cd VoxelTTO
conda create -n voxeltto python=3.10 -y
conda activate voxeltto
pip install -r requirements.txt
```

Install the differentiable renderer based on stochastic volumetric rendering
```bash
cd Geometry-Grounded-Gaussian-Splatting/submodules/diff-gaussian-rasterization
pip install . --no-build-isolation
```


## Running the Code
Run all commands below from the repository root with the `voxeltto`
environment activated:

```bash
cd ~/autodl-tmp/VoxelTTO
conda activate voxeltto
```

### 1. Prepare the data

The scene directory (Colmap pose) is expected to contain:

```text
office0_50_500/
├── images/                 # input RGB images
└── sparse/
    └── gt/                 # COLMAP-format camera priors used by TTO
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt    # \
```

The default inference configuration runs test-time camera optimization (TTO),
so `sparse/gt/cameras.txt` and `sparse/gt/images.txt` are required.

### 2. Run inference


```bash
python demo_colmap.py \
  --scene_dir $data_dir \
  --shared_camera \
  --conf_thres_value 0.0 \
  --use_training_resolution_crop \
  --align_gt_camera \
  --config_file configs/inference.yaml \
  --checkpoint_path $checkpoint_path \
  --sparse_subdir $output_dir_name
```


Useful options:

- Add `--bf16` to reduce backbone memory usage on supported GPUs.
- Change `--sparse_subdir` to select a different output directory name.
- Run `python demo_colmap.py --help` for the complete argument list.

