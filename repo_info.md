# Final Assignment — Repository Info

## Project Overview

This project implements a **3D Object Detection system for Radar point clouds** using the **CenterPoint** architecture. It detects three classes — **Car, Pedestrian, Cyclist** — on the **View of Delft (VoD)** dataset. The codebase is built on **PyTorch Lightning**, configured with **Hydra** YAML files, and optionally logged via **Weights & Biases**.

---

## Directory Structure

```
final_assignment/
├── final_assignment.ipynb        # Notebook for interactive experimentation
├── data/
│   └── view_of_delft/            # VoD dataset (radar point clouds, labels, calibration)
├── src/
│   ├── __init__.py
│   ├── config/                   # Hydra YAML configs
│   │   ├── train.yaml
│   │   ├── eval.yaml
│   │   ├── test.yaml
│   │   └── model/
│   │       ├── centerpoint.yaml        # LiDAR variant (4-channel input)
│   │       └── centerpoint_radar.yaml  # Radar variant (7-channel input)
│   ├── dataset/                  # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── view_of_delft.py      # VoD Dataset class
│   │   └── utils.py              # Collate function & helpers
│   ├── evaluation/               # Evaluation utils (VoD eval API)
│   │   └── __init__.py
│   ├── model/                    # Model components
│   │   ├── __init__.py
│   │   ├── detector/
│   │   │   └── centerpoint.py          # Main detector (Lightning module)
│   │   ├── voxel_encoders/
│   │   │   ├── pillar_encoder.py       # PillarFeatureNet
│   │   │   └── utils.py               # Voxel encoder helpers
│   │   ├── middle_encoders/
│   │   │   └── pillar_scatter.py       # PointPillarsScatter
│   │   ├── backbones/
│   │   │   └── second.py              # SECOND backbone
│   │   ├── necks/
│   │   │   └── second_fpn.py          # SECONDFPN neck
│   │   ├── heads/
│   │   │   └── centerpoint_head.py    # CenterHead (detection head)
│   │   ├── losses/
│   │   │   ├── gaussian_focal_loss.py # Heatmap classification loss
│   │   │   ├── l1_loss.py             # Bbox regression loss
│   │   │   └── losses_utils.py        # Loss utilities
│   │   ├── bricks/
│   │   │   └── conv_module.py         # ConvModule building block
│   │   └── utils/
│   │       ├── base_box3d.py          # BaseInstance3DBoxes
│   │       ├── lidar_box3d.py         # LiDARInstance3DBoxes
│   │       ├── base_points.py         # BasePoints
│   │       ├── box3d_utils.py         # 3D box utilities
│   │       ├── centerpoint_bbox_coders.py  # BBox encoding/decoding
│   │       └── utils_func.py          # General utility functions
│   ├── ops/                      # Custom CUDA/C++ operations
│   │   ├── voxelize.py               # Hard/dynamic voxelization
│   │   ├── scatter_points.py         # DynamicScatter
│   │   ├── iou3d.py                  # BEV IoU & NMS
│   │   ├── points_in_boxes.py        # Point-in-box tests
│   │   └── cpp_pkgs/                 # C++/CUDA kernel sources & setup.py
│   └── tools/                    # Entry-point scripts
│       ├── train.py                  # Training script
│       ├── eval.py                   # Validation evaluation script
│       ├── test.py                   # Test-set inference script
│       ├── zip_files.py              # Package results for leaderboard
│       └── slurm_train.sh            # SLURM cluster launcher
```

---

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING / INFERENCE PIPELINE                   │
│                                                                         │
│  Raw Radar Points (N, 5): x, y, z, RCS, Doppler velocity               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │  Voxelization │  (src/ops/voxelize.py)                               │
│  │  Layer         │  Points → Voxels (M, max_pts, C) + Coordinates      │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────────┐                                                   │
│  │ PillarFeatureNet  │  (src/model/voxel_encoders/pillar_encoder.py)    │
│  │ (Voxel Encoder)   │  Voxels → Pillar features (M, 64)               │
│  └──────┬───────────┘                                                   │
│         ▼                                                               │
│  ┌─────────────────────┐                                                │
│  │ PointPillarsScatter  │  (src/model/middle_encoders/pillar_scatter.py)│
│  │ (Middle Encoder)     │  Scatter pillars → BEV grid (B, 64, H, W)    │
│  └──────┬──────────────┘                                                │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ SECOND        │  (src/model/backbones/second.py)                     │
│  │ Backbone      │  3-stage conv with stride-2 → multi-scale features   │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ SECONDFPN     │  (src/model/necks/second_fpn.py)                     │
│  │ Neck          │  Upsample & concatenate → (B, 384, H/4, W/4)        │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ CenterHead    │  (src/model/heads/centerpoint_head.py)               │
│  │ (Det. Head)   │  3 per-class task heads → heatmaps + bbox regression │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────────────┐                                               │
│  │ BBoxCoder / NMS       │  (src/model/utils/centerpoint_bbox_coders.py)│
│  │ Post-processing       │  Decode predictions → final 3D detections    │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Descriptions

### Entry Points (`src/tools/`)

| File | Purpose |
|------|---------|
| `train.py` | Main training script. Loads Hydra config, builds dataset & model, runs PyTorch Lightning `Trainer.fit()`. |
| `eval.py` | Loads a trained checkpoint and runs validation evaluation, reporting AP metrics. |
| `test.py` | Runs inference on the test set and saves predictions in KITTI format. |
| `zip_files.py` | Packages prediction results into a ZIP file for leaderboard submission. |
| `slurm_train.sh` | SLURM job script to launch distributed training on a cluster. |

### Dataset (`src/dataset/`)

| File | Purpose |
|------|---------|
| `view_of_delft.py` | `ViewOfDelft` dataset class. Loads radar point clouds (`.bin` files), parses KITTI-format labels, transforms bounding boxes from camera → LiDAR coordinates, and returns `(points, gt_labels, gt_bboxes_3d, meta_info)`. |
| `utils.py` | `collate_vod_batch()` — custom collate function to handle variable-length objects per sample. |

### Model Components (`src/model/`)

| File | Purpose |
|------|---------|
| `detector/centerpoint.py` | **CenterPoint** — the main `LightningModule`. Orchestrates the full forward pass: voxelization → encoder → backbone → neck → head. Implements `training_step`, `validation_step`, `test_step`, and assembles all submodules from config. |
| `voxel_encoders/pillar_encoder.py` | **PillarFeatureNet** — encodes raw voxelized points into pillar features. Augments each point with cluster-center offsets and voxel-center offsets, then applies a series of linear layers. |
| `voxel_encoders/utils.py` | Helper functions for the voxel encoder (e.g., `get_paddings_indicator`). |
| `middle_encoders/pillar_scatter.py` | **PointPillarsScatter** — takes pillar features (M, 64) and their voxel coordinates, scatters them into a dense 2D BEV feature map (B, 64, H, W). |
| `backbones/second.py` | **SECOND** backbone — multi-stage 2D convolutional network with stride-2 downsampling at each stage. Produces multi-scale feature maps. |
| `necks/second_fpn.py` | **SECONDFPN** — Feature Pyramid Network that upsamples and concatenates multi-scale backbone outputs into a single feature tensor (B, 384, H', W'). |
| `heads/centerpoint_head.py` | **CenterHead** — anchor-free detection head. Creates 3 independent task-specific sub-heads (one per class: Car, Pedestrian, Cyclist). Each task predicts a heatmap + regression outputs (offset, height, dimensions, rotation). Also handles training target generation and inference-time box decoding via `BBoxCoder`. |
| `bricks/conv_module.py` | **ConvModule** — reusable building block wrapping Conv2d + BatchNorm + ReLU, used throughout the backbone and neck. |

### Losses (`src/model/losses/`)

| File | Purpose |
|------|---------|
| `gaussian_focal_loss.py` | **GaussianFocalLoss** — focal-loss variant for heatmap classification, weighting positive/negative pixels differently based on Gaussian-splat targets. |
| `l1_loss.py` | **L1Loss** — smooth L1 regression loss for bounding box parameters (offset, size, rotation). |
| `losses_utils.py` | Utility functions: `weight_reduce_loss`, `weighted_loss` decorator. |

### Model Utilities (`src/model/utils/`)

| File | Purpose |
|------|---------|
| `base_box3d.py` | `BaseInstance3DBoxes` — abstract 3D bounding box container with shared operations (volume, rotation, flipping, etc.). |
| `lidar_box3d.py` | `LiDARInstance3DBoxes` — LiDAR-coordinate 3D box representation (x, y, z, w, l, h, yaw). Gravity center at geometric center. |
| `base_points.py` | `BasePoints` — point cloud container with coordinate transforms. |
| `box3d_utils.py` | Standalone 3D box helper functions (rotation matrices, corner computation, etc.). |
| `centerpoint_bbox_coders.py` | **CenterPointBBoxCoder** — encodes ground-truth boxes into grid-relative targets and decodes network outputs back into 3D boxes during inference. |
| `utils_func.py` | General utilities: `bias_init_with_prob`, `kaiming_init`, `multi_apply`, `clip_sigmoid`. |

### Custom CUDA Operations (`src/ops/`)

| File | Purpose |
|------|---------|
| `voxelize.py` | `Voxelization` — converts a raw point cloud into a voxel grid (hard voxelization). Backed by custom CUDA kernels for speed. |
| `scatter_points.py` | `DynamicScatter` — mean/max-reduces point features within each voxel, yielding one feature vector per voxel. |
| `iou3d.py` | BEV IoU computation and **NMS** (Non-Maximum Suppression) for removing duplicate detections at inference time. |
| `points_in_boxes.py` | Determines which points fall inside which 3D boxes — used during training target assignment. |
| `cpp_pkgs/` | C++ and CUDA source files compiled via `setup.py` to create the native extensions above. |

### Configs (`src/config/`)

| File | Key Settings |
|------|-------------|
| `train.yaml` | `epochs=50`, `batch_size=2`, optimizer=`AdamW(lr=0.001, wd=0.01)`, dataset paths, WandB logging |
| `eval.yaml` | Checkpoint path, validation dataset split, evaluation parameters |
| `test.yaml` | Checkpoint path, test dataset split, output directory for predictions |
| `model/centerpoint_radar.yaml` | Radar model: 7 input channels, voxel_size=[0.32, 0.32, 5], max 5 pts/voxel, max 8000/20000 voxels (train/val), all layer dimensions |
| `model/centerpoint.yaml` | LiDAR model: 4 input channels, voxel_size=[0.16, 0.16, 4], max 32 pts/voxel, max 16000/40000 voxels |

---

## Key Relationships & Data Flow

### Training Flow

```
train.py
  │
  ├─ Hydra loads: train.yaml + model/centerpoint_radar.yaml
  │
  ├─ Instantiates ViewOfDelft dataset (src/dataset/view_of_delft.py)
  │     └─ Returns: (points, gt_labels, gt_bboxes_3d, meta_info)
  │
  ├─ DataLoader with collate_vod_batch (src/dataset/utils.py)
  │
  ├─ Instantiates CenterPoint detector (src/model/detector/centerpoint.py)
  │     ├─ Voxelization layer      (src/ops/voxelize.py)
  │     ├─ PillarFeatureNet         (src/model/voxel_encoders/pillar_encoder.py)
  │     ├─ PointPillarsScatter      (src/model/middle_encoders/pillar_scatter.py)
  │     ├─ SECOND backbone          (src/model/backbones/second.py)
  │     ├─ SECONDFPN neck           (src/model/necks/second_fpn.py)
  │     └─ CenterHead               (src/model/heads/centerpoint_head.py)
  │           ├─ GaussianFocalLoss  (src/model/losses/gaussian_focal_loss.py)
  │           └─ L1Loss             (src/model/losses/l1_loss.py)
  │
  └─ PyTorch Lightning Trainer.fit()
        ├─ training_step: forward → loss → backward
        └─ validation_step: forward → decode → KITTI-format → VoD eval
```

### Inference Flow

```
test.py / eval.py
  │
  ├─ Load checkpoint → CenterPoint model
  │
  ├─ For each sample:
  │     points → Voxelize → PillarFeatureNet → Scatter → Backbone → Neck → CenterHead
  │                                                                          │
  │                                                      ┌──────────────────┘
  │                                                      ▼
  │                                              Top-K on heatmaps
  │                                              BBoxCoder.decode()
  │                                              Circle NMS (iou3d.py)
  │                                              LiDAR → Camera coord transform
  │                                              Filter by image bounds
  │                                                      │
  │                                                      ▼
  │                                              Final 3D Detections
  │
  └─ Save predictions in KITTI format / compute AP metrics
```

### Import Dependency Graph

```
detector/centerpoint.py
  ├── imports → ops/voxelize.py
  ├── imports → voxel_encoders/pillar_encoder.py
  │                 └── imports → bricks/conv_module.py
  │                 └── imports → voxel_encoders/utils.py
  ├── imports → middle_encoders/pillar_scatter.py
  ├── imports → backbones/second.py
  │                 └── imports → bricks/conv_module.py
  ├── imports → necks/second_fpn.py
  │                 └── imports → bricks/conv_module.py
  ├── imports → heads/centerpoint_head.py
  │                 ├── imports → losses/gaussian_focal_loss.py
  │                 │                 └── imports → losses/losses_utils.py
  │                 ├── imports → losses/l1_loss.py
  │                 │                 └── imports → losses/losses_utils.py
  │                 ├── imports → utils/centerpoint_bbox_coders.py
  │                 └── imports → utils/utils_func.py
  ├── imports → utils/lidar_box3d.py
  │                 └── imports → utils/base_box3d.py
  │                                  └── imports → utils/box3d_utils.py
  └── imports → ops/iou3d.py (for NMS)

dataset/view_of_delft.py
  ├── imports → utils/lidar_box3d.py
  └── imports → utils/base_points.py

tools/train.py
  ├── imports → dataset/view_of_delft.py
  ├── imports → dataset/utils.py
  └── imports → model/detector/centerpoint.py

tools/eval.py
  ├── imports → dataset/view_of_delft.py
  ├── imports → dataset/utils.py
  └── imports → model/detector/centerpoint.py

tools/test.py
  ├── imports → dataset/view_of_delft.py
  ├── imports → dataset/utils.py
  └── imports → model/detector/centerpoint.py
```

---

## Key Hyperparameters (Radar Config)

| Category | Parameter | Value |
|----------|-----------|-------|
| **Input** | Channels | 7 (x, y, z, RCS, v_r + 2 augmented) |
| **Voxel** | Voxel size | [0.32, 0.32, 5] meters |
| **Voxel** | Point cloud range | [0, -25.6, -3, 51.2, 25.6, 2] |
| **Voxel** | Max points per voxel | 5 |
| **Voxel** | Max voxels (train / val) | 8,000 / 20,000 |
| **Backbone** | Stages | 3 |
| **Backbone** | Layers per stage | [3, 5, 5] |
| **Backbone** | Output channels | [64, 128, 256] |
| **Head** | Input channels | 384 |
| **Head** | Task heads | 3 (Car, Pedestrian, Cyclist) |
| **Loss** | Heatmap loss weight | 1.0 |
| **Loss** | BBox loss weight | 0.25 |
| **Training** | Optimizer | AdamW |
| **Training** | Learning rate | 0.001 |
| **Training** | Weight decay | 0.01 |
| **Training** | Batch size | 2 |
| **Training** | Epochs | 50 |

---

## Summary

The pipeline follows the standard **pillar-based 3D detection** paradigm:

1. **Data** — `ViewOfDelft` loads sparse radar point clouds and KITTI-format ground truth.
2. **Voxelization** — Points are binned into a fixed grid of pillars (vertical columns).
3. **Encoding** — `PillarFeatureNet` produces a feature vector per pillar; `PointPillarsScatter` projects them onto a dense Bird's Eye View (BEV) feature map.
4. **Feature extraction** — `SECOND` backbone extracts multi-scale 2D features; `SECONDFPN` fuses them into a single tensor.
5. **Detection** — `CenterHead` predicts a Gaussian heatmap (object centers) and regresses bounding box attributes per class.
6. **Post-processing** — Top-K selection, `BBoxCoder` decoding, and circle-NMS yield final 3D detections.
7. **Evaluation** — Predictions are formatted in KITTI style and scored with the VoD evaluation API (Average Precision).
