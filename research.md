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

### 3.1 Additional high-upside papers that were not in the original main list

If the goal shifts from "incrementally improve the current baseline" to "study papers with higher performance ceilings and stronger novelty", the following papers are worth adding to the reading list as well.

| Paper | Venue / Year | Core idea | Why it matters for a performance-first project |
|---|---|---|---|
| **Multi-class Road User Detection with 3+1D Radar in the View-of-Delft Dataset** | RA-L 2022 | Establishes VoD and studies how elevation, Doppler, RCS, and temporal accumulation affect 3+1D radar detection. | This is still the most important grounding paper for VoD. It is not the flashiest method, but it tells you which radar signals actually matter and why multi-sweep aggregation is fundamental rather than optional. |
| **HGSFusion** | AAAI 2025 / arXiv 2024 | Uses a **Radar Hybrid Generation Module (RHGM)** to generate denser hybrid radar points from DOA-aware PDFs, plus a **Dual Sync Module (DSM)** to do spatial and modality synchronization. | High practical upside. On VoD, it surpasses the previous SOTA by **+2.65 EAA AP** and **+6.53 RoI AP**; on TJ4DRadSet, it reaches **37.21 3D mAP / 43.23 BEV mAP**, improving BEV AP by **+2.03** over the prior best. This is one of the strongest performance-first radar-camera papers to study. |
| **RICCARDO** | CVPR 2025 | Predicts an explicit **radar hit distribution** conditioned on monocular object properties, then uses that predicted distribution as a convolution kernel for radar matching before fusion refinement. | This is one of the most original fusion papers in the recent literature because it does not treat radar-camera fusion as a black box. It reaches **0.695 NDS / 0.630 mAP** on nuScenes test and **0.622 NDS / 0.544 mAP** on val, outperforming prior radar-camera baselines built on the same detector family. |
| **RCTrans** | AAAI 2025 / arXiv 2024 | Introduces a **Radar Dense Encoder** to densify sparse radar tokens and a **Pruning Sequential Decoder** to progressively align radar and image tokens with queries. | This is a strong candidate if you are willing to move beyond dense CenterPoint-style heads. It reports **59.4 NDS / 52.0 mAP** on nuScenes val and **64.7 NDS / 57.8 mAP** on test, exceeding RCBEVDet and CRN on the same benchmark. |
| **TARS** | ICCV 2025 | Jointly performs object detection and radar scene flow, and introduces a **Traffic Vector Field (TVF)** to model motion rigidity at the traffic level instead of the instance level. | Not a detector in the narrow sense, but a very important motion paper. It improves radar scene-flow benchmarks by **23%** on a proprietary dataset and **15%** on VoD, which matters if you want a deeper temporal model than simple sweep accumulation. |
| **4D-RaDiff** | arXiv 2025 | Uses **latent diffusion** to generate object-level and scene-level 4D radar point clouds for augmentation, annotation transfer, and pretraining. | This is the most data-centric high-upside paper in the set. It reports consistent detection gains from synthetic radar augmentation and claims pretraining on synthetic data can reduce the amount of required annotated radar data by **up to 90%** while retaining comparable detection performance. |
| **AsyncBEV** | arXiv 2026 | Adds a lightweight BEV-space flow alignment module to handle sensor asynchrony between modalities. | This matters if your long-term system becomes multi-sensor and temporal. Under a **0.5 s** offset, it improves dynamic-object NDS by **+16.6%** for CMT and **+11.9%** for UniBEV, so it is a strong robustness paper even if not the first paper to implement. |
| **DAT++** | arXiv 2023 | Builds a deformable multi-head attention mechanism with data-dependent key/value locations, preserving global context while avoiding dense ViT attention everywhere. | This is not an autonomous-driving radar paper, but it is useful architectural inspiration if you want a stronger BEV transformer or query decoder. It reports **85.9%** ImageNet accuracy, **54.5 / 47.0** COCO instance segmentation mAP, and **51.5** ADE20K mIoU. |
| **DeforHMR** | 3DV 2025 / arXiv 2024 | Uses a **query-agnostic deformable cross-attention** decoder on top of a frozen ViT encoder. | This is also cross-domain inspiration rather than a direct detector paper. The main reason to read it is the decoder design: the deformable cross-attention idea may transfer to radar BEV decoding or cross-modal token fusion. |

One additional 2026 direction worth keeping on a watchlist is **SGE-Flow: 4D mmWave Radar 3D Object Detection via Spatiotemporal Geometric Enhancement and Inter-Frame Flow**. From currently circulating summaries, it appears to combine velocity-displacement compensation, distribution-aware density restoration, and a transformer-based inter-frame flow module. That makes it conceptually relevant to your project, but I have not yet verified an official paper page, so I would treat it as a provisional lead rather than a top-priority cited source.

### 3.2 Performance-first recommendation: five extra papers to prioritize

If you want to seriously study only five additional papers, and the priority is **maximum performance ceiling rather than implementation ease**, I would recommend the following order.

| Year | Venue | Paper | Link | Core idea | Why prioritize it for performance |
|---|---|---|---|---|---|
| 2025 | **ICCV 2025** | **CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection** | [link](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_CVFusion_Cross-View_Fusion_of_4D_Radar_and_Camera_for_3D_ICCV_2025_paper.html) | Proposes a **two-stage cross-view fusion** pipeline: Stage 1 uses **Radar Guided Iterative BEV Fusion** to generate high-recall proposals, and Stage 2 refines them with heterogeneous point, image, and BEV features through **Point-Guided Fusion (PGF)** and **Grid-Guided Fusion (GGF)**. | This is the strongest pure performance paper in the allowed set for a VoD-style project. The paper reports a **+9.10% mAP** gain over prior SOTA on **VoD** and **+3.68% mAP** on **TJ4DRadSet**, while your existing note already records **65.4 mAP** on VoD val. |
| 2025 | **CVPR 2025** | **RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion** | [link](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html) | Introduces a **query-based radar-camera fusion transformer** with adaptive circular query initialization, a **radar-guided depth head**, and an **implicit dynamic catcher** to strengthen BEV temporal representation. | High-ceiling and directly relevant to your benchmark setting. It reports **64.9 mAP / 70.2 NDS** on **nuScenes test**, and the paper states **54.4 mAP** on the full VoD annotated area and **78.6 mAP** in the driving corridor, ranking first on VoD. |
| 2025 | **CVPR 2025** | **RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection** | [link](https://openaccess.thecvf.com/content/CVPR2025/html/Long_RICCARDO_Radar_Hit_Prediction_and_Convolution_for_Camera-Radar_3D_Object_CVPR_2025_paper.html) | Explicitly predicts a **radar-hit distribution** conditioned on monocular object properties, then uses the predicted distribution as a convolution kernel to match real radar points and refine object positions. | Methodologically the most distinctive fusion paper in the allowed set. It reaches **0.695 NDS / 0.630 mAP** on the **nuScenes test set** and **0.622 NDS / 0.544 mAP** on the **validation set**, outperforming several prior radar-camera baselines. |
| 2025 | **ICCV 2025** | **RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion** | [link](https://openaccess.thecvf.com/content/ICCV2025/html/Bang_RCTDistill_Cross-Modal_Knowledge_Distillation_Framework_for_Radar-Camera_3D_Object_Detection_ICCV_2025_paper.html) | Combines temporal fusion with three distillation modules: **RAKD** for range-azimuth-aware transfer, **TKD** for temporal alignment with dynamic objects, and **RDKD** for foreground-background relational distillation. | Very strong if you optimize purely for benchmark performance and your rules allow LiDAR as a training-only teacher. The paper reports **state-of-the-art radar-camera fusion performance on both nuScenes and VoD** and a **26.2 FPS** inference speed; the abstract also highlights **+4.7 mAP / +4.9 NDS** over the student model. |
| 2025 | **ICCV 2025** | **DoppDrive: Doppler-Driven Temporal Aggregation for Improved Radar Object Detection** | [link](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | Uses **Doppler-driven temporal aggregation** before detection: historical points are shifted according to their dynamic Doppler component, and each point gets an adaptive aggregation duration to reduce both radial and tangential scatter. | Best performance-oriented radar-only choice among the allowed papers. It is detector-agnostic and the project page reports large AP gains across multiple datasets and detectors, for example **81.7 -> 89.1 AP** on aiMotive with SMURF and **92.5 -> 95.8 AP** on Radial with SMURF compared with standard aggregation. |

If you only have time to go deep on **two** of them, I would start with **CVFusion** and **RICCARDO**. The first is the strongest VoD-style performance paper in the allowed pool, and the second gives you the most distinctive fusion formulation.

## 4. What I would try first, in priority order

The section above is the **performance-first reading list**. The implementation roadmap below is still the more pragmatic route for this exact repository.

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
- Multi-class Road User Detection with 3+1D Radar in the View-of-Delft Dataset. IEEE Robotics and Automation Letters, 2022. https://research.tudelft.nl/en/publications/multi-class-road-user-detection-with-31d-radar-in-the-view-of-del/
- CenterPoint Transformer for BEV Object Detection with Automotive Radar. CVPRW 2024. https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Saini_CenterPoint_Transformer_for_BEV_Object_Detection_with_Automotive_Radar_CVPRW_2024_paper.html
- RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Bang_RadarDistill_Boosting_Radar-based_Object_Detection_Performance_via_Knowledge_Distillation_from_CVPR_2024_paper.html
- Bootstrapping Autonomous Driving Radars with Self-Supervised Learning. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Hao_Bootstrapping_Autonomous_Driving_Radars_with_Self-Supervised_Learning_CVPR_2024_paper.html
- HGSFusion: Radar-Camera Fusion with Hybrid Generation and Synchronization for 3D Object Detection. AAAI 2025 / arXiv 2024. https://arxiv.org/abs/2412.11489
- RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html
- RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Long_RICCARDO_Radar_Hit_Prediction_and_Convolution_for_Camera-Radar_3D_Object_CVPR_2025_paper.html
- RCTrans: Radar-Camera Transformer via Radar Densifier and Sequential Decoder for 3D Object Detection. AAAI 2025 / arXiv 2024. https://arxiv.org/abs/2412.12799
- AttentiveGRU: Recurrent Spatio-Temporal Modeling for Advanced Radar-Based BEV Object Detection. CVPRW 2025. https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Saini_AttentiveGRU_Recurrent_Spatio-Temporal_Modeling_for_Advanced_Radar-Based_BEV_Object_Detection_CVPRW_2025_paper.html
- DoppDrive: Doppler-Driven Temporal Aggregation for Improved Radar Object Detection. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html
- DoppDrive project page. https://yuvalhg.github.io/DoppDrive/
- CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_CVFusion_Cross-View_Fusion_of_4D_Radar_and_Camera_for_3D_ICCV_2025_paper.html
- TARS: Traffic-Aware Radar Scene Flow Estimation. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Wu_TARS_Traffic-Aware_Radar_Scene_Flow_Estimation_ICCV_2025_paper.html
- 4D-RaDiff: Latent Diffusion for 4D Radar Point Cloud Generation. arXiv 2025. https://arxiv.org/abs/2512.14235
- RadarNeXt: lightweight and real-time 3D object detector based on 4D mmWave imaging radar. Journal on Advances in Signal Processing, 2025. https://link.springer.com/article/10.1186/s13634-025-01271-2
- Graph Query Networks for Object Detection with Automotive Radar. WACV 2026 / arXiv 2025. https://arxiv.org/abs/2511.15271
- AsyncBEV: Cross-modal Flow Alignment in Asynchronous 3D Object Detection. arXiv 2026. https://arxiv.org/abs/2601.12994
- DAT++: Spatially Dynamic Vision Transformer with Deformable Attention. arXiv 2023. https://arxiv.org/abs/2309.01430
- DeforHMR: Vision Transformer with Deformable Cross-Attention for 3D Human Mesh Recovery. 3DV 2025 / arXiv 2024. https://arxiv.org/abs/2411.11214
- RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/papers/Bang_RCTDistill_Cross-Modal_Knowledge_Distillation_Framework_for_Radar-Camera_3D_Object_Detection_ICCV_2025_paper.pdf
