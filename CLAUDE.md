# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D object detection on radar point clouds using CenterPoint architecture on the View of Delft (VoD) dataset. Detects Car, Pedestrian, and Cyclist classes. Built with PyTorch Lightning + Hydra configs + Weights & Biases logging. This is a TU Delft course assignment (RO47020 Advanced Machine Perception) where the goal is to improve the radar baseline detector.

## Commands

All commands must be run from the repository root. Scripts use Hydra for config — override any YAML parameter via CLI (e.g., `batch_size=4 epochs=12`).

```bash
# Train (default: centerpoint_radar model, 50 epochs, batch_size=2)
python src/tools/train.py exp_id=centerpoint_radar_baseline batch_size=4 num_workers=2 epochs=12

# Evaluate on validation set
python src/tools/eval.py checkpoint_path=PATH_TO_CKPT

# Test set inference (for leaderboard submission)
python src/tools/test.py checkpoint_path=PATH_TO_CKPT

# Package test predictions for leaderboard upload
python src/tools/zip_files.py --res_folder outputs/EXP_ID/test_preds --output_path outputs/EXP_ID/submission.zip
```

Build CUDA extensions (required once before training):
```bash
cd src/ops/cpp_pkgs && python setup.py develop && cd ../../..
```

## Architecture

The pipeline is **pillar-based CenterPoint** (anchor-free detection):

```
Raw Points → Voxelization → PillarFeatureNet → PointPillarsScatter → SECOND backbone → SECONDFPN neck → CenterHead → NMS → 3D Detections
```

The main orchestrator is `src/model/detector/centerpoint.py` — a `LightningModule` that wires all submodules and implements `training_step`/`validation_step`/`test_step`.

**Key data flow details:**
- Radar input: 5 raw channels (x, y, z, RCS, Doppler velocity), augmented to 7 channels by PillarFeatureNet (cluster-center & voxel-center offsets)
- Voxelization produces pillars on a BEV grid; scatter creates dense 2D feature map
- CenterHead has 3 independent task heads (one per class), each predicting heatmap + regression (offset, height, dims, rotation)
- Post-processing: top-K on heatmaps → BBoxCoder decode → circle NMS → LiDAR-to-camera transform → image bounds filter

## Config System

Hydra configs in `src/config/`. Two model variants:
- `src/config/model/centerpoint_radar.yaml` — **radar** (7 input channels, voxel_size=[0.32, 0.32, 5], 160×160 BEV grid)
- `src/config/model/centerpoint.yaml` — **LiDAR** (4 input channels, voxel_size=[0.16, 0.16, 5], 320×320 BEV grid)

Train config defaults to `centerpoint_radar`. Override model with `model=centerpoint` for LiDAR variant.

Outputs go to `outputs/{exp_id}/` with checkpoints in `checkpoints/` subdirectory. Best model selected by `validation/entire_area/mAP`.

## Constraints

- Only radar and monocular camera data allowed (no stereo, no LiDAR)
- No additional pip dependencies beyond the existing `amp` conda environment
- Training limited to 4 hours on a shared A100 GPU on DelftBlue
- Dataset lives at `data/view_of_delft/` (symlinked on DelftBlue from `/projects/ro47020/view_of_delft_PUBLIC`)
- Custom CUDA ops in `src/ops/` wrap C++/CUDA kernels built via `src/ops/cpp_pkgs/setup.py`

## Dataset

VoD dataset in KITTI format. `ViewOfDelft` class in `src/dataset/view_of_delft.py` loads `.bin` point clouds and parses KITTI labels. Camera-to-LiDAR coordinate transforms happen during loading. Custom `collate_vod_batch` handles variable-length GT objects per sample.

Splits: `train`, `val`, `test`. Validation runs at batch_size=1. Validation/test use the KITTI eval API via `src/evaluation/`.
