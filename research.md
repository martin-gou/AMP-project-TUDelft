# Research: How to Improve the Current AMP Radar 3D Detection Pipeline

Date: 2026-03-11

This note analyzes the current repository and maps recent radar / radar-camera 3D detection literature to concrete, code-level improvement directions for this project.

## 1. What the current repository actually implements

The current codebase is a clean radar-only CenterPoint baseline:

- `src/dataset/view_of_delft.py` loads one frame at a time and returns only `radar_data`. The `sequential_loading` flag exists but is not used, so I infer the model is effectively **single-frame radar only**.
- `src/config/model/centerpoint_radar.yaml` uses PointPillars-style encoding with `voxel_size=[0.32, 0.32, 5]`, `max_num_points=5`, `feat_channels=[64]`, and a standard `SECOND + SECONDFPN + CenterHead`.
- `src/model/heads/centerpoint_head.py` predicts `reg / height / dim / rot`; there is no enabled velocity head in the current config.
- `src/tools/train.py` uses plain AdamW and does not configure a scheduler, EMA, class-balanced sampling, or any explicit data augmentation.
- There is no camera branch, no temporal branch, no recurrent memory, and no radar-specific clutter suppression module.

In short, your baseline is:

1. single-frame
2. radar-only
3. generic pillar encoder
4. generic BEV backbone
5. minimal training recipe

For LiDAR, this would be a reasonable baseline. For radar, it leaves too much performance on the table.

## 2. Main limitations of the current baseline

### 2.1 Temporal information is completely unused

Radar is sparse. Multi-frame accumulation is one of the most consistently effective ways to improve detection quality, especially for small and distant objects. Your current dataset code loads only the current frame, even though the assignment allows temporal data and GNSS/IMU.

Consequence:

- weak long-range detection
- unstable pedestrian / cyclist recall
- underuse of Doppler

### 2.2 Doppler is present but not modeled explicitly

The pipeline feeds radar point features into a standard PFN-style pillar encoder, but there is no dedicated Doppler-aware aggregation, motion head, or temporal compensation module. For radar, this is a major miss.

Consequence:

- moving-object features remain noisy
- naive temporal accumulation would create motion smear
- the model cannot learn motion-consistent object features well

### 2.3 The encoder is not radar-specific

The current `PillarFeatureNet` is essentially the PointPillars recipe with minimal adaptation. Recent radar papers repeatedly show that radar benefits from:

- global context
- clutter suppression
- RCS-aware processing
- temporal feature fusion
- query-based or instance-aware refinement

Consequence:

- background clutter and sparse reflections are not handled aggressively enough
- small objects are especially vulnerable

### 2.4 No camera fusion

The assignment explicitly allows monocular camera input, but the repository does not use it. Recent VoD results suggest that camera-radar fusion is the largest remaining upside if you can afford the engineering effort.

### 2.5 Training recipe is much weaker than current literature

I do not see augmentation, CBGS/class rebalancing, LR scheduling, temporal training, or stronger post-processing. Even before architectural changes, this is low-hanging fruit.

## 3. Recent papers most relevant to this repo

I focused on recent, primary sources that are close to your setting: radar point clouds, VoD, BEV detection, temporal modeling, Doppler, and radar-camera fusion.

| Paper | Venue / Year | Why it matters for your code |
|---|---|---|
| **RCBEVDet** | CVPR 2024 | Very relevant because it reports on **VoD** and shows that a **radar-specific encoder matters**, not just fusion. On VoD val it reports **49.99 mAP** for the entire area and **69.80 mAP** for ROI. Its ablations show that the proposed radar backbone adds about **+3.0 mAP** and the RCS-aware BEV encoder adds about **+1.9 mAP**. |
| **CenterPoint Transformer for BEV Object Detection with Automotive Radar** | CVPRW 2024 | Closest architectural relative to your current baseline. It keeps the center-based detection idea but injects **global transformer context** into center predictions. Reported **+23.6% car mAP** over the best radar-only submission on nuScenes. Strong evidence that your current `SECOND + FPN` stack is too local for radar. |
| **RadarDistill** | CVPR 2024 | Shows that radar BEV representations can be improved by training-time distillation from LiDAR features. It reports **20.5 mAP / 43.7 NDS** on nuScenes radar-only. This is potentially powerful, but only if your course rules allow LiDAR as a training-only teacher. |
| **Bootstrapping Autonomous Driving Radars with Self-Supervised Learning** | CVPR 2024 | Shows that radar representation quality can improve via SSL pretraining, with **+5.8% mAP** downstream improvement. Good idea if you have access to large unlabeled radar data or raw radar heatmaps. |
| **RaCFormer** | CVPR 2025 | Strong radar-camera paper with explicit relevance to **VoD**. It uses **query-based cross-view fusion**, a **radar-guided depth head**, and an **implicit dynamic catcher** to model temporal information in BEV. It reports **54.4 mAP** on the entire annotated area and **78.6 mAP** on the VoD driving corridor, and ranked first on VoD. |
| **AttentiveGRU** | CVPRW 2025 | Strong evidence for **recurrent temporal BEV modeling** in radar. It introduces an attention-based recurrent module and reports a **21% increase in car mAP** over the best radar-only submission on nuScenes. This is highly relevant if you want to stay radar-only. |
| **DoppDrive** | ICCV 2025 | Extremely relevant to your data format: it is a **detector-agnostic Doppler-driven temporal aggregation** method applied before detection. It improves AP across multiple detectors and datasets by using Doppler to reduce motion smear during multi-frame accumulation. This is one of the best papers for a practical next step in your repo. |
| **CVFusion** | ICCV 2025 | A very strong **two-stage radar-camera fusion** paper. It reports **+9.10 mAP improvement on VoD** over previous SOTA and achieves **65.4 mAP** on VoD val. The main lesson is important: BEV fusion alone is not enough; **proposal refinement with multi-view instance features** gives a big second-stage gain. |
| **RadarNeXt** | Journal on Advances in Signal Processing, 2025 | Good radar-only reference for a stronger but still efficient backbone. It proposes a re-parameterizable backbone and a **Multi-path Deformable Foreground Enhancement Network** for clutter suppression. On **VoD five-scan** it reports **50.48 mAP** at **67 FPS** on RTX A4000. |
| **Graph Query Networks for Object Detection with Automotive Radar** | WACV 2026 / arXiv 2025 | Newer radar-only direction. It replaces plain grid reasoning with **graph-query relational reasoning** and reports up to **+8.2 mAP** over the strongest prior radar method on nuScenes. High upside, but also higher implementation risk for your repo. |

## 4. What I would try first, in priority order

### Priority 1: Add temporal radar aggregation

This is the single most important radar-only improvement for your repo.

Why:

- your code is currently single-frame
- radar sparsity is the dominant problem
- the assignment allows temporal data and GNSS/IMU
- recent papers repeatedly show temporal modeling is high value

What to implement first:

1. Load `K` past radar sweeps in `src/dataset/view_of_delft.py`.
2. Ego-motion compensate past sweeps into the current frame.
3. Add a relative timestamp channel `dt`.
4. Start with simple point-level accumulation.
5. Then add **Doppler-aware compensation** inspired by DoppDrive.

Best low-risk variant:

- 3-sweep or 5-sweep aggregation
- channels like `[x, y, z, RCS, v_r, v_ra, t, dt]` if available
- no architecture change at first

Better second variant:

- add a lightweight **ConvGRU / AttentiveGRU-style recurrent block** after `middle_encoder` or after the first neck stage

Why this should help your repo:

- your current pipeline already produces BEV features; temporal fusion in BEV is an easy fit
- this gives a strong radar-only gain without needing camera engineering

### Priority 2: Strengthen the training recipe before rewriting the model

This is the cheapest improvement bucket and should be done early.

What is missing now:

- no explicit geometric augmentation
- no class-balanced group sampling
- no LR scheduler
- no temporal training
- no tuned post-processing sweeps

Recommended changes:

1. Add global BEV flip, rotation, and scaling augmentations.
2. Add radar point dropout / jitter / point masking.
3. Add class-balanced sampling or at least per-class oversampling.
4. Add cosine decay or OneCycle schedule.
5. Sweep `score_threshold`, `nms_thr`, `post_max_size`, and `min_radius`.
6. Try smaller `voxel_size` in XY for better pedestrian/cyclist localization.

Expected outcome:

- easy baseline lift
- cleaner comparison for later architecture ablations

### Priority 3: Replace the generic pillar encoder with a radar-specific encoder

Your current encoder is too simple for radar.

Strong directions from the literature:

- **RCBEVDet**: dual-stream radar backbone + RCS-aware BEV encoding
- **CenterPoint Transformer**: global context into center features
- **RadarNeXt**: clutter-aware foreground enhancement and deformable fusion
- **GQN**: object-centric relational reasoning

Practical upgrade path for this repo:

1. Keep the current detector/head but replace `PillarFeatureNet` with a stronger radar encoder.
2. Add one of:
   - local attention within pillars
   - RCS-aware weighting
   - deformable conv block after scatter
   - transformer block on BEV features
3. Increase encoder capacity:
   - more than one PFN layer
   - larger feature dimension
   - more voxels / finer XY resolution

My recommendation:

- first try a **small transformer/deformable block on BEV features**
- only later redesign the full radar backbone

This gives a better cost/benefit ratio than rewriting the entire pipeline immediately.

### Priority 4: Use Doppler explicitly, not just as another input channel

Right now Doppler is likely just mixed with other raw point features. Recent papers suggest that is not enough.

Good options:

1. Add a velocity or motion-consistency auxiliary head.
2. Add Doppler-aware temporal fusion weights.
3. Separate static vs dynamic feature aggregation paths.
4. Use Doppler to suppress bad historical points during accumulation.

Closest paper guidance:

- **DoppDrive** for pre-detection temporal alignment
- **RaCFormer** for dynamic-aware BEV modeling
- **CenterPoint Transformer** for auxiliary velocity prediction

### Priority 5: Add camera-radar fusion if you want the biggest absolute gain

If the goal is leaderboard performance rather than the cleanest radar-only story, this is probably the highest upside.

Why:

- camera is allowed by the assignment
- recent VoD leaders are radar-camera methods
- RCBEVDet, RaCFormer, and CVFusion all show strong gains

Recommended fusion order:

1. **Late BEV fusion baseline**
   - image encoder
   - image-to-BEV lift/splat
   - fuse with radar BEV before the head
2. **Radar-guided depth**
   - use radar to improve image-to-BEV lifting
3. **Query-based or two-stage refinement**
   - RaCFormer / CVFusion style

My view:

- if you have limited time, do not start with full query-based fusion
- build a simple BEV fusion baseline first, then refine

### Priority 6: Distillation and pretraining are strong, but conditional

These are good ideas, but only under the right constraints.

#### 6.1 LiDAR teacher distillation

Evidence:

- RadarDistill
- RCTDistill (ICCV 2025) for radar-camera + temporal fusion

Use this only if:

- the course staff explicitly permits LiDAR as a **training-only teacher**

If this is not clearly allowed, skip it.

#### 6.2 Self-supervised radar pretraining

Evidence:

- Bootstrapping Autonomous Driving Radars with Self-Supervised Learning

Use this only if:

- you have access to the right raw radar representation or large unlabeled radar corpora

For this repo, this is lower priority than temporal aggregation and camera fusion.

## 5. Concrete experiment roadmap for this repository

### Phase A: Cheap and high-signal experiments

Goal: improve the current baseline without large refactors.

1. Add geometric augmentation.
2. Add LR scheduler.
3. Tune voxel size, max voxels, and post-processing thresholds.
4. Add class rebalancing.

Why first:

- low engineering cost
- establishes a stronger baseline
- makes later ablations fairer

### Phase B: Strong radar-only path

Goal: get a real radar-specific gain while staying close to the current code.

1. Multi-sweep loading
2. Ego-motion alignment
3. `dt` channel
4. Doppler-aware aggregation
5. BEV temporal fusion block

This is the path I would recommend if you want a solid radar-only scientific story in the paper.

### Phase C: Highest-upside path

Goal: chase leaderboard performance.

1. Add monocular camera branch
2. Build a BEV fusion baseline
3. Add radar-guided depth
4. Add proposal/query refinement

This is more work, but it aligns best with the strongest VoD papers from 2024-2025.

## 6. Suggested code touchpoints

If I were implementing the above in this repo, I would start here:

- `src/dataset/view_of_delft.py`
  - add multi-frame loading
  - add timestamp / sweep index
  - add ego-motion compensation
  - optionally expose image paths / camera tensors

- `src/config/model/centerpoint_radar.yaml`
  - temporal settings
  - finer voxel size
  - larger encoder/backbone options
  - optional velocity head settings

- `src/model/voxel_encoders/pillar_encoder.py`
  - RCS-aware weighting
  - Doppler-aware feature branches
  - stronger local attention / PFN depth

- `src/model/detector/centerpoint.py`
  - temporal aggregation
  - BEV recurrent block
  - optional camera branch

- `src/model/heads/centerpoint_head.py`
  - velocity / motion auxiliary head
  - query refinement or second-stage refinement

- `src/tools/train.py`
  - scheduler
  - EMA
  - stronger logging for ablations
  - balanced sampling if implemented at loader level

## 7. My ranked recommendation for your next experiments

If you want the best return on effort, I would run experiments in this order:

1. **Upgrade the training recipe**: augmentation + scheduler + threshold tuning.
2. **Add 3-sweep or 5-sweep radar accumulation**.
3. **Make the accumulation Doppler-aware**.
4. **Add a lightweight temporal BEV module**.
5. **Upgrade the encoder with a small transformer/deformable block**.
6. **Only then decide between radar-only paper story vs camera-radar fusion push**.

If your goal is maximum performance rather than the cleanest ablation story:

1. strong baseline
2. temporal radar
3. camera-radar BEV fusion
4. query / proposal refinement

## 8. Bottom line

The current repository is a good **teaching baseline**, but by 2025 standards it is clearly underpowered for radar.

The three biggest gaps are:

1. **no temporal modeling**
2. **no explicit Doppler-aware reasoning**
3. **no camera fusion**

For this exact codebase, the best next step is not a full rewrite. The best next step is:

1. strengthen the training recipe
2. add multi-frame radar aggregation with ego-motion and Doppler handling
3. add a lightweight temporal BEV fusion block

After that, decide whether you want:

- a strong radar-only method with a clean scientific story, or
- a higher-ceiling camera-radar fusion method for leaderboard chasing

## 9. Sources

- RCBEVDet: Radar-camera Fusion in Bird’s Eye View for 3D Object Detection. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Lin_RCBEVDet_Radar-camera_Fusion_in_Birds_Eye_View_for_3D_Object_CVPR_2024_paper.html
- CenterPoint Transformer for BEV Object Detection with Automotive Radar. CVPRW 2024. https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Saini_CenterPoint_Transformer_for_BEV_Object_Detection_with_Automotive_Radar_CVPRW_2024_paper.html
- RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Bang_RadarDistill_Boosting_Radar-based_Object_Detection_Performance_via_Knowledge_Distillation_from_CVPR_2024_paper.html
- Bootstrapping Autonomous Driving Radars with Self-Supervised Learning. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Hao_Bootstrapping_Autonomous_Driving_Radars_with_Self-Supervised_Learning_CVPR_2024_paper.html
- RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html
- AttentiveGRU: Recurrent Spatio-Temporal Modeling for Advanced Radar-Based BEV Object Detection. CVPRW 2025. https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Saini_AttentiveGRU_Recurrent_Spatio-Temporal_Modeling_for_Advanced_Radar-Based_BEV_Object_Detection_CVPRW_2025_paper.html
- DoppDrive: Doppler-Driven Temporal Aggregation for Improved Radar Object Detection. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html
- CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_CVFusion_Cross-View_Fusion_of_4D_Radar_and_Camera_for_3D_ICCV_2025_paper.html
- RadarNeXt: lightweight and real-time 3D object detector based on 4D mmWave imaging radar. Journal on Advances in Signal Processing, 2025. https://link.springer.com/article/10.1186/s13634-025-01271-2
- Graph Query Networks for Object Detection with Automotive Radar. WACV 2026 / arXiv 2025. https://arxiv.org/abs/2511.15271
- RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/papers/Bang_RCTDistill_Cross-Modal_Knowledge_Distillation_Framework_for_Radar-Camera_3D_Object_Detection_ICCV_2025_paper.pdf
