# Experiment Plan

## Goal

Improve the final assignment metric `Driving_corridor_area_mAP`, which matches `validation/ROI/mAP` in this repo.

My scope:
- architecture improvements
- camera input / fusion

Not my scope:
- data augmentation


## Core Rules

1. Change one idea at a time.
2. Always compare against the same strong baseline.
3. Rank experiments by `validation/ROI/mAP` first, not `entire_area/mAP`.
4. Do a short smoke test before every full run.
5. Do not combine two new modules before each one is proven useful alone.
6. Every full run must append one row to `report.md`.


## Fixed Baseline Protocol

Before any ablation, freeze a baseline after the coordinate fix.

- Dataset: current radar-in-LiDAR-coordinate version
- Main metric: `validation/ROI/mAP`
- Secondary metric: `validation/entire_area/mAP`
- Keep train split / val split unchanged
- Keep teammate's augmentation setting fixed while running my ablations

Suggested baseline tag:
- `baseline_coordfix`


## Decision Rule

Use the following rule after each full run:

- `keep`: ROI mAP improves clearly and no major class collapses
- `maybe`: ROI mAP improves only slightly; rerun once or inspect class-wise ROI AP
- `drop`: ROI mAP does not improve, or only entire-area mAP improves while ROI stays flat/down

Practical interpretation:
- strong keep: ROI mAP gain is clearly above noise
- weak keep: small ROI gain but consistent gains in at least 2 classes
- reject: worse ROI, unstable training, or much higher cost without clear gain


## Standard Workflow For Every Experiment

### Stage A: smoke test

Purpose: catch shape bugs, projection bugs, OOM, and broken training.

Suggested command pattern:

```bash
python src/tools/train.py exp_id=EXP_NAME epochs=3 val_every=1 batch_size=2
```

Check:
- training starts normally
- validation finishes
- checkpoint is saved
- `report.md` gets a row after training
- ROI mAP is logged

### Stage B: full run

Only after smoke test passes.

Suggested command pattern:

```bash
python src/tools/train.py exp_id=EXP_NAME epochs=12 batch_size=4 num_workers=2
```

Use your normal full budget after the short run looks valid.


## Recommended Experiment Order

### Phase 0: lock a clean baseline

#### E0 - Baseline after coordinate fix

Hypothesis:
- coordinate consistency alone gives a cleaner baseline

Files:
- `src/dataset/view_of_delft.py`
- `src/tools/train.py`

Goal:
- establish the reference ROI mAP for all later work

Decision:
- do not move on until this run is stable and recorded in `report.md`


### Phase 1: radar-only architecture improvements

This phase should come first. It is lower risk than camera fusion and gives a strong radar-only contribution for the report.

#### E1 - Deeper pillar encoder

Hypothesis:
- the current pillar encoder is too weak for sparse radar features

Change:
- increase PFN depth from one layer to two layers

Files:
- `src/config/model/centerpoint_radar.yaml`
- `src/model/voxel_encoders/pillar_encoder.py`

Minimal version:
- change `feat_channels` from `[64]` to `[64, 64]`

Why first:
- small code change
- low compute increase
- directly targets radar feature extraction

Keep if:
- ROI mAP improves over E0


#### E2 - Add point distance feature

Hypothesis:
- explicit distance helps radar points because sparsity and reliability vary strongly with range

Change:
- enable `with_distance`

Files:
- `src/config/model/centerpoint_radar.yaml`
- `src/model/voxel_encoders/pillar_encoder.py`

Important:
- compare against E1, not against a mixed bundle of new changes

Keep if:
- ROI mAP improves over E1, or class-wise ROI AP becomes more balanced


#### E3 - Radar-aware channel gating in pillar encoder

Hypothesis:
- RCS and Doppler features are useful, but not equally useful for all points

Change:
- add a lightweight gating block or SE-style channel attention over point features before PFN aggregation

Files:
- `src/model/voxel_encoders/pillar_encoder.py`

Implementation note:
- keep it light; do not build a heavy transformer here

Keep if:
- ROI mAP improves over the best of E1/E2


#### E4 - Residual BEV backbone

Hypothesis:
- the current `SECOND` backbone is too plain; residual blocks may improve feature reuse and optimization

Change:
- replace plain conv stacks in the BEV backbone with residual blocks

Files:
- `src/model/backbones/second.py`
- maybe `src/config/model/centerpoint_radar.yaml`

Important:
- keep channels the same first
- do not add attention in the same experiment

Keep if:
- ROI mAP improves over the best radar front-end model


#### E5 - Residual backbone plus lightweight attention

Hypothesis:
- after residual blocks work, a small SE/CBAM-style attention may further improve feature quality

Change:
- add lightweight attention inside the residual BEV backbone

Files:
- `src/model/backbones/second.py`

Important:
- only run this if E4 already helped

Keep if:
- ROI mAP improves over E4 with acceptable runtime increase


### Phase 2: camera fusion

Only start this phase after choosing the best radar-only model from Phase 1.

Recommended base:
- best model among E1-E5 by ROI mAP


#### E6 - Camera projection debug only

Hypothesis:
- before adding camera features, projection and coordinate handling must be verified

Change:
- load image in the dataset
- project radar points to image using LiDAR-to-camera transform

Files:
- `src/dataset/view_of_delft.py`
- `src/dataset/utils.py`

Important:
- radar points are now in LiDAR coordinates, so projection must use LiDAR-to-camera transforms
- do not use raw radar-frame projection anymore

Success criteria:
- projected points align visually with image content on a few samples
- no training run needed yet


#### E7 - Frozen pretrained ResNet-18 image encoder

Hypothesis:
- camera features can add semantics cheaply if extracted from a pretrained network

Change:
- add a pretrained `ResNet-18` image encoder
- keep it frozen in the first version

Files:
- add new camera backbone module under `src/model/backbones/`
- update `src/model/detector/centerpoint.py`
- update `src/dataset/view_of_delft.py`
- update `src/dataset/utils.py`

Why this version:
- lowest risk camera baseline
- cheap enough for the 4h budget

Keep if:
- fusion run improves ROI mAP over the best radar-only model


#### E8 - Point-level camera fusion (recommended first fusion method)

Hypothesis:
- the most practical way to use camera is to sample image features at radar point projections and append them to point features before voxelization / pillar encoding

Change:
- project each radar point into the image
- sample image feature at that pixel
- append or gated-fuse a low-dimensional image feature to the radar point feature

Files:
- `src/dataset/view_of_delft.py`
- `src/dataset/utils.py`
- `src/model/detector/centerpoint.py`
- `src/model/voxel_encoders/pillar_encoder.py`
- `src/config/model/centerpoint_radar.yaml`

Recommended first design:
- ResNet-18
- frozen image backbone
- use one late feature map only
- reduce image feature dimension before concatenation

Why this is the best first camera idea:
- fits the current radar-first pipeline
- low extra compute because radar points are sparse
- much easier than dense image-to-BEV fusion

Keep if:
- ROI mAP improves over E7 or over the best radar-only model


#### E9 - Partial fine-tuning of the image encoder

Hypothesis:
- once frozen camera fusion works, fine-tuning later image stages may improve domain adaptation to VoD

Change:
- unfreeze only later ResNet stages

Files:
- new camera backbone module
- `src/model/detector/centerpoint.py`

Recommended first setting:
- unfreeze only the last one or two ResNet stages

Do not do first:
- full fine-tuning from the start
- ResNet-50 or larger before ResNet-18 works

Keep if:
- ROI mAP improves over E8 without making training unstable or too slow


#### E10 - Stronger image backbone only if needed

Hypothesis:
- if point-level fusion works, a slightly stronger image encoder may help

Change:
- upgrade `ResNet-18` to `ResNet-34`

Important:
- only after E8/E9 already show positive results
- do not jump to big backbones too early

Keep if:
- gain is worth the extra cost


## Update After The Previous Runs

结束了前面的测试，决定在 `radar_pfn2_gate` 的基础上继续做实验。

Reason:
- `radar_pfn2_gate` is still the strongest clean radar-first line by ROI mAP.
- `E4` and `E5` did not beat it clearly enough to justify a heavier BEV backbone.
- `E8` and `E9` showed that the current camera fusion path is not helping.
- The next experiments should therefore stay radar-only and focus on the pillar front-end, where the gain actually appeared.


### Phase 3: `radar_pfn2_gate` follow-up

Base model:
- `centerpoint_radar_pfn2_gate`


#### E3_1 - Add distance feature on top of `radar_pfn2_gate`

Hypothesis:
- explicit point distance may complement the current gate because radar reliability changes strongly with range

Change:
- keep the current two-layer PFN and channel gate
- enable `with_distance`

Files:
- `src/config/model/centerpoint_radar_pfn2_gate_e3_1.yaml`
- `src/tools/slurm_train_e3_1.sh`

Important:
- do not combine this with a new gate type
- compare directly against `radar_pfn2_gate`

Keep if:
- ROI mAP improves over `radar_pfn2_gate`


#### E3_2 - Replace channel gate with point-wise reliability gate

Hypothesis:
- a point-wise gate may work better than a channel gate because radar noise is often point-specific inside a pillar

Change:
- replace the current channel gate with a point-wise reliability gate before PFN aggregation
- keep voxelization and backbone unchanged

Files:
- `src/model/voxel_encoders/pillar_encoder.py`
- `src/config/model/centerpoint_radar_pfn2_gate_e3_2.yaml`
- `src/tools/slurm_train_e3_2.sh`

Important:
- keep `with_distance` off in this experiment
- compare directly against `radar_pfn2_gate`

Keep if:
- ROI mAP improves over `radar_pfn2_gate`
- class-wise ROI AP becomes more stable for `Pedestrian` and `Cyclist`


#### E3_3 - Finer voxelization for the gated PFN baseline

Hypothesis:
- the current 0.32 m grid may be too coarse for the ROI metric, especially for small objects

Change:
- keep the current gated PFN design
- use a finer BEV voxel size
- slightly increase pillar capacity to avoid dropping too many points/voxels

Files:
- `src/config/model/centerpoint_radar_pfn2_gate_e3_3.yaml`
- `src/tools/slurm_train_e3_3.sh`

Important:
- update scatter output shape, grid size, and bbox coder voxel size consistently
- do not change the BEV backbone in the same run

Keep if:
- ROI mAP improves over `radar_pfn2_gate`
- runtime remains acceptable for the current Slurm budget


## What I Recommend First

If time is limited, do this exact order:

1. E0 baseline
2. E1 deeper pillar encoder
3. E2 distance feature
4. E3 radar-aware gating
5. E4 residual BEV backbone
6. E6 camera projection debug
7. E7 frozen ResNet-18 encoder
8. E8 point-level camera fusion
9. E9 partial fine-tune

After finishing the original sequence above, continue with this follow-up order:

1. E3_1 distance feature on `radar_pfn2_gate`
2. E3_2 point-wise reliability gate
3. E3_3 finer voxelization on the gated PFN baseline


## What I Do Not Recommend As A First Attempt

- training a ResNet from scratch on VoD
- using ResNet as a direct replacement for the whole 3D detector backbone
- building a separate monocular detector branch first
- dense image-to-BEV fusion before simple point-level fusion works
- changing head, backbone, encoder, and fusion all at once


## Why The ResNet Idea Is Good, But Only In The Right Place

Good use of ResNet:
- as a pretrained camera feature extractor
- first frozen, then partially fine-tuned

Not recommended:
- as the only main architecture idea
- as a from-scratch image model trained only on VoD

Best practical choice:
- `ResNet-18` first
- `ResNet-34` only if the first fusion version already works


## Suggested Experiment Naming

Use explicit `exp_id` names so `report.md` stays readable.

Examples:
- `baseline_coordfix`
- `radar_pfn2`
- `radar_pfn2_dist`
- `radar_pfn2_dist_gate`
- `radar_resbackbone`
- `cam_res18_frozen`
- `cam_res18_pointfusion`
- `cam_res18_pointfusion_ftlast`


## What To Check In `report.md`

After each run, verify that `report.md` records:

- best ROI mAP
- best ROI checkpoint path
- best entire-area mAP
- best entire-area checkpoint path
- run directory

Use this file as the main ranking list.


## Simple Template For Manual Notes

For each experiment, add a short note outside `report.md` if needed:

- hypothesis
- exact code diff
- smoke test status
- full run status
- keep / maybe / drop

This is useful because `report.md` stores results, but not the reasoning.


## Final Deliverable Strategy

Best report story if things go well:

1. Coordinate-consistent radar baseline
2. Radar-aware architecture improvement
3. Camera semantic fusion on top of the improved radar model
4. Clear ablation table showing what helped and what did not

This gives both:
- a solid engineering improvement
- a clean scientific narrative for the paper
