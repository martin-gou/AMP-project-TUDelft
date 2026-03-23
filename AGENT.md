# AGENT.md

## Mission
This repository is a TU Delft Advanced Machine Perception final assignment.
The goal is to improve a radar-based CenterPoint 3D detector on the View of Delft (VoD) dataset.
Allowed inputs for the final method are radar and optional monocular camera only.
LiDAR and stereo camera are not allowed.

## Deliverables / grading context
- Main deliverables: code zip, leaderboard submission, WandB logs, and a scientific report.
- The report core must be exactly 4 pages excluding references.
- Performance is judged by improvement over the provided radar baseline on the VoD leaderboard.
- Code changes should stay reproducible and respect the official train/val/test splits.

## Environment constraints
- Do not add dependencies beyond the existing `amp` conda environment.
- Intended runtime is DelftBlue `gpu-a100-small`: 1 A100, 2 CPUs, 4h wall time.
- Common module/env setup:
  - `module load 2024r1 miniconda3/4.12.0 cuda/12.5`
  - `conda activate amp`
- Build custom CUDA ops before training:
  - `cd src/ops/cpp_pkgs && python setup.py develop`

## Main entry points
- Train:
  - `python src/tools/train.py exp_id=centerpoint_radar_baseline batch_size=4 num_workers=2 epochs=12`
- Evaluate on validation set:
  - `python src/tools/eval.py checkpoint_path=PATH_TO_CKPT`
- Run test-set inference:
  - `python src/tools/test.py checkpoint_path=PATH_TO_CKPT`
- Zip predictions for submission:
  - `python src/tools/zip_files.py --res_folder outputs/EXP_ID/test_preds --output_path outputs/EXP_ID/submission.zip`

## Actual runtime behavior
- Training uses `src/config/train.yaml`, which defaults to `src/config/model/centerpoint_radar.yaml`.
- Checkpoints are selected by `validation/entire_area/mAP`, not ROI/driving-corridor mAP.
- `eval.py` and `test.py` both reuse `trainer.validate(...)`; there is no dedicated `test_step`.
- `test.py` switches:
  - `model.inference_mode = 'test'`
  - `model.save_results = True`
- Test predictions are therefore written through validation hooks.
- Validation/test effectively require batch size 1 because `CenterPoint.validation_step()` asserts it.

## Core architecture
Pipeline:
`radar points -> Voxelization -> PillarFeatureNet -> PointPillarsScatter -> SECOND -> SECONDFPN -> CenterHead -> BBox decode -> circle NMS -> KITTI-format predictions`

Key files:
- `src/model/detector/centerpoint.py`: Lightning module and end-to-end orchestration
- `src/dataset/view_of_delft.py`: VoD radar dataset loading and GT conversion
- `src/dataset/utils.py`: collate function used by all scripts
- `src/model/voxel_encoders/pillar_encoder.py`: pillar feature encoding
- `src/model/middle_encoders/pillar_scatter.py`: BEV scatter
- `src/model/backbones/second.py`: backbone
- `src/model/necks/second_fpn.py`: FPN neck
- `src/model/heads/centerpoint_head.py`: CenterHead and decoding/loss logic

## Baseline model facts
- Classes:
  - `Car`
  - `Pedestrian`
  - `Cyclist`
- Radar point cloud range:
  - `[0, -25.6, -3, 51.2, 25.6, 2]`
- Radar voxel size:
  - `[0.32, 0.32, 5]`
- Max points per voxel:
  - `5`
- Max voxels:
  - train `8000`
  - val/test `20000`
- Backbone output channels:
  - `[64, 128, 256]`
- Neck output channels:
  - `384` total
- Detection head:
  - 3 one-class task heads, one each for Car / Pedestrian / Cyclist

## Dataset and box conventions
- `ViewOfDelft` loads radar points from `vod_frame_data.radar_data`, but returns them under the misleading key `lidar_data`.
- `collate_vod_batch()` renames this to `pts`.
- Split files and GT labels are still read from the `data/view_of_delft/lidar/...` tree. This is intentional in the current code.
- GT locations are transformed from camera coordinates to LiDAR coordinates.
- GT dimensions are reordered from KITTI `h,w,l` to internal `l,w,h`.
- Internal boxes use `LiDARInstance3DBoxes(origin=(0.5, 0.5, 0))`, i.e. bottom-centered boxes in the VoD LiDAR convention.
- During export, predictions are converted back to KITTI text format and dimensions are written as `h,w,l`.

## Important gotchas
- Docs describe raw radar as 5-D (`x,y,z,RCS,Doppler`), but `src/config/model/centerpoint_radar.yaml` sets `voxel_encoder.in_channels: 7`.
- Before editing radar features, verify the real runtime shape of `vod_frame_data.radar_data`.
- The code contains explicit `.cuda()` calls, so it is GPU-only in practice.
- `src/config/eval.yaml` and `src/config/test.yaml` default to the LiDAR config (`centerpoint`), but checkpoint loading usually overrides the model structure.
- `src/tools/test.py` hardcodes `devices=1`.
- `src/config/train.yaml` contains a typo: `class_namses`. It appears unused.
- Leaderboard emphasis is driving-corridor/ROI mAP, but training saves top checkpoints by entire-area mAP.
- The best validation checkpoint may therefore not be the best leaderboard checkpoint.
- Post-processing is tightly coupled to VoD camera geometry and image bounds; be careful when changing transforms, yaw handling, or result formatting.

## Safe places to modify for improvements
- Radar feature engineering / temporal accumulation:
  - `src/dataset/view_of_delft.py`
  - `src/dataset/utils.py`
  - `src/model/voxel_encoders/pillar_encoder.py`
- Backbone / neck / head experiments:
  - `src/config/model/centerpoint_radar.yaml`
  - `src/model/backbones/second.py`
  - `src/model/necks/second_fpn.py`
  - `src/model/heads/centerpoint_head.py`
- Post-processing / test-time tricks:
  - `src/model/detector/centerpoint.py`
  - `src/model/heads/centerpoint_head.py`
- Camera fusion is allowed only with monocular images and must preserve all assignment rules.

## Working rules for future edits
- Never introduce LiDAR or stereo-camera information into the final method.
- Never train on validation/test or alter the official splits.
- Prefer small, reproducible ablations because GPU time is capped at 4 hours.
- If touching dataset format, box conventions, or result writing, always verify that validation still produces correct KITTI-format outputs.
- Submission zip must contain prediction files only, not the enclosing folder.
