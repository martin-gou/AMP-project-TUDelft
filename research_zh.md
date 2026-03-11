# 研究报告：如何提升当前 AMP 雷达 3D 检测 Pipeline 的性能

日期：2026-03-11

这份文档基于当前仓库的实现，对现有 pipeline 进行分析，并结合近年的雷达 / 雷达-相机 3D 检测论文，整理出适合这个项目的、可直接落地的性能提升方向。

## 1. 当前仓库实际实现了什么

当前代码库本质上是一个比较标准的 radar-only CenterPoint baseline：

- `src/dataset/view_of_delft.py` 每次只加载单帧数据，并且只返回 `radar_data`。虽然代码里有 `sequential_loading` 参数，但实际上并没有真正使用，所以我判断当前模型是一个**单帧雷达模型**。
- `src/config/model/centerpoint_radar.yaml` 使用的是典型 PointPillars 风格编码：`voxel_size=[0.32, 0.32, 5]`，`max_num_points=5`，`feat_channels=[64]`，后面接 `SECOND + SECONDFPN + CenterHead`。
- `src/model/heads/centerpoint_head.py` 当前预测的是 `reg / height / dim / rot`，配置里没有启用 velocity head。
- `src/tools/train.py` 只用了最基础的 AdamW，没有 scheduler、EMA、class-balanced sampling，也没有看到显式的数据增强。
- 整个工程里没有 camera 分支、没有 temporal 分支、没有 recurrent memory，也没有专门的 radar clutter suppression 模块。

也就是说，你当前的 baseline 可以概括为：

1. 单帧
2. 仅雷达
3. 通用的 pillar encoder
4. 通用的 BEV backbone
5. 很基础的训练策略

如果这是一个 LiDAR baseline，这样的配置还算合理；但对 radar 来说，它明显没有把潜力挖出来。

## 2. 当前 baseline 的主要限制

### 2.1 完全没有利用时序信息

Radar 的先天问题是点云非常稀疏。对 radar 检测来说，多帧累积是最稳定、最常见、也最有效的提升手段之一，尤其对远距离目标和小目标非常关键。你当前的数据集代码只加载当前帧，虽然作业允许使用 temporal data 和 GNSS/IMU，但这部分完全没被用起来。

直接后果：

- 远距离目标检测弱
- Pedestrian / Cyclist 的召回不稳定
- Doppler 信息没有被真正发挥作用

### 2.2 虽然输入里有 Doppler，但没有显式建模

现在的 pipeline 只是把 radar 点特征丢进一个 PFN 风格的 pillar encoder，但并没有 Doppler-aware aggregation、motion head 或者 temporal compensation 这类模块。对 radar 来说，这个缺口很大。

直接后果：

- 运动目标特征容易被噪声污染
- 如果直接做多帧累积，会产生 motion smear
- 模型学不到稳定的运动一致性特征

### 2.3 编码器并不是 radar-specific 的

当前 `PillarFeatureNet` 基本还是 PointPillars 的思路，只做了有限改动。而近年的 radar 论文反复说明，radar 更依赖：

- 全局上下文
- clutter suppression
- RCS-aware processing
- temporal feature fusion
- query-based 或 instance-aware refinement

直接后果：

- 背景杂波和稀疏反射没有被很好抑制
- 小目标尤其容易掉点

### 2.4 没有使用 camera fusion

作业明确允许 monocular camera，但当前仓库完全没有使用图像信息。近两年在 VoD 上的结果已经说明，如果工程时间允许，camera-radar fusion 往往是最大的性能上限来源。

### 2.5 训练策略明显落后于当前主流做法

我没有看到 augmentation、CBGS / class rebalancing、LR scheduling、temporal training 或更强的 post-processing。哪怕先不改结构，这部分也有明显的低成本增益空间。

## 3. 与当前仓库最相关的近期论文

我优先挑选了和你当前 setting 最接近的一手来源：radar point clouds、VoD、BEV detection、temporal modeling、Doppler 以及 radar-camera fusion。

| 论文 | 会议 / 年份 | 为什么和你的代码最相关 |
|---|---|---|
| **RCBEVDet** | CVPR 2024 | 这篇和你的场景非常接近，因为它报告了 **VoD** 上的结果，并且明确说明 **radar-specific encoder 本身就重要**，不仅仅是 fusion。它在 VoD val 上报告了 **49.99 mAP**（entire area）和 **69.80 mAP**（ROI）。它的消融还显示，提出的 radar backbone 带来约 **+3.0 mAP**，RCS-aware BEV encoder 再带来约 **+1.9 mAP**。 |
| **CenterPoint Transformer for BEV Object Detection with Automotive Radar** | CVPRW 2024 | 这是和你当前 baseline 架构最近的一篇。它保留了 center-based detection 的思想，但往 center prediction 里加入了 **global transformer context**。论文报告其在 nuScenes 上相对最佳 radar-only 提交带来 **+23.6% car mAP**，说明你当前 `SECOND + FPN` 的局部感受野对 radar 来说不够。 |
| **RadarDistill** | CVPR 2024 | 这篇说明可以通过 LiDAR 特征蒸馏来提升 radar BEV 表征。它在 nuScenes radar-only 上报告了 **20.5 mAP / 43.7 NDS**。这个方向可能很强，但前提是课程规则允许把 LiDAR 作为训练阶段 teacher。 |
| **Bootstrapping Autonomous Driving Radars with Self-Supervised Learning** | CVPR 2024 | 这篇说明 radar 表征可以通过自监督预训练显著提升，下游任务可获得 **+5.8% mAP**。如果你手头有大量未标注 radar 数据或者更原始的 radar 表示，它会有价值。 |
| **RaCFormer** | CVPR 2025 | 非常强的 radar-camera 论文，而且明确和 **VoD** 有关。它用了 **query-based cross-view fusion**、**radar-guided depth head** 和一个用于建模时序信息的 **implicit dynamic catcher**。论文报告在 VoD 上达到 **54.4 mAP**（entire annotated area）和 **78.6 mAP**（driving corridor），并且排名第一。 |
| **AttentiveGRU** | CVPRW 2025 | 这篇对 radar-only 路线非常有参考价值。它引入 attention-based recurrent module 做 **时序 BEV 建模**，报告在 nuScenes 上相对最佳 radar-only 提交带来 **21% car mAP 提升**。如果你希望保持 radar-only，这是很值得借鉴的方向。 |
| **DoppDrive** | ICCV 2025 | 这篇和你的数据形式高度相关，因为它做的是 **detector-agnostic 的 Doppler-driven temporal aggregation**，也就是在检测前先利用 Doppler 做时间对齐与融合。论文在多个 detector 和数据集上都提升了 AP，核心价值在于减少多帧累积带来的 motion smear。对你这个仓库来说，这是最实用的下一步之一。 |
| **CVFusion** | ICCV 2025 | 一篇很强的 **两阶段 radar-camera fusion** 工作。它在 VoD 上相对之前 SOTA 带来了 **+9.10 mAP**，最终在 VoD val 达到 **65.4 mAP**。这篇论文最关键的结论是：只做 BEV fusion 还不够，**proposal refinement + multi-view instance feature** 的第二阶段增益很大。 |
| **RadarNeXt** | Journal on Advances in Signal Processing, 2025 | 这是一个兼顾性能和效率的 radar-only 参考。它提出了可重参数化 backbone 和 **Multi-path Deformable Foreground Enhancement Network** 用于 clutter suppression。在 **VoD five-scan** 上报告了 **50.48 mAP**，速度为 **67 FPS**（RTX A4000）。 |
| **Graph Query Networks for Object Detection with Automotive Radar** | WACV 2026 / arXiv 2025 | 更新一点的 radar-only 路线。它不再单纯依赖规则网格，而是引入 **graph-query relational reasoning**。论文报告在 nuScenes 上相对先前强方法最高可带来 **+8.2 mAP**。上限高，但对你当前仓库来说实现风险也更大。 |

## 4. 我建议优先尝试什么

### 优先级 1：加入时序 radar 聚合

这是对你当前 radar-only 仓库最重要的一步。

原因：

- 当前代码是单帧
- radar 稀疏性是最核心问题
- 作业允许 temporal data 和 GNSS/IMU
- 近年的 radar 论文几乎都说明时序建模非常值

第一步怎么做：

1. 在 `src/dataset/view_of_delft.py` 里加载过去 `K` 帧 radar sweep。
2. 用 ego-motion 把历史帧对齐到当前帧。
3. 加一个相对时间通道 `dt`。
4. 先做最简单的 point-level accumulation。
5. 然后再加入受 DoppDrive 启发的 **Doppler-aware compensation**。

低风险版本：

- 先试 3-sweep 或 5-sweep aggregation
- 如果数据里支持，可以把通道扩成 `[x, y, z, RCS, v_r, v_ra, t, dt]`
- 第一版先不改网络结构

更进一步的版本：

- 在 `middle_encoder` 后，或者 neck 第一层后，加一个轻量级 **ConvGRU / AttentiveGRU 风格的 recurrent block**

为什么这很适合你的仓库：

- 你当前 pipeline 已经在生成 BEV feature 了，所以把 temporal fusion 放在 BEV 空间很自然
- 这样可以在不引入 camera 工程复杂度的情况下，先拿到一波很扎实的 radar-only 提升

### 优先级 2：在大改结构之前，先补齐训练策略

这部分是成本最低、最值得先做的提升项。

当前缺的东西：

- 没有显式几何增强
- 没有 class-balanced group sampling
- 没有 LR scheduler
- 没有时序训练
- 没有系统化 post-processing sweep

建议修改：

1. 加 BEV 视角下的 global flip、rotation、scaling augmentation。
2. 加 radar point dropout / jitter / point masking。
3. 加 class-balanced sampling，至少也要做 per-class oversampling。
4. 加 cosine decay 或 OneCycle scheduler。
5. sweep `score_threshold`、`nms_thr`、`post_max_size`、`min_radius`。
6. 尝试更小的 XY `voxel_size`，提升 Pedestrian / Cyclist 的定位精度。

预期效果：

- 可以先把 baseline 抬高
- 后续做结构消融时，对比更干净

### 优先级 3：把通用 pillar encoder 升级成 radar-specific encoder

当前 encoder 对 radar 来说还是太弱了。

论文里最值得借鉴的方向有：

- **RCBEVDet**：dual-stream radar backbone + RCS-aware BEV encoding
- **CenterPoint Transformer**：在 center feature 上引入 global context
- **RadarNeXt**：clutter-aware foreground enhancement + deformable fusion
- **GQN**：object-centric relational reasoning

对你这个仓库来说，更实际的升级路径是：

1. 先保留当前 detector/head，不急着全改。
2. 先把 `PillarFeatureNet` 换成更强的 radar encoder。
3. 可以优先加以下其中一种：
   - pillar 内部的 local attention
   - RCS-aware weighting
   - scatter 之后接 deformable conv block
   - 在 BEV feature 上加 transformer block
4. 适度增大 encoder 容量：
   - 不只用一层 PFN
   - 提高 feature dimension
   - 提高 max voxels 或细化 XY 分辨率

我的建议是：

- 第一版先尝试 **在 BEV features 上加一个小型 transformer / deformable block**
- 不要一开始就重写整个 radar backbone

这样更符合投入产出比。

### 优先级 4：显式利用 Doppler，而不是把它当普通输入通道

现在 Doppler 很可能只是和其他 raw point feature 一起混在 encoder 里。近年的论文基本都说明，这样不够。

可以尝试的方式：

1. 增加 velocity 或 motion-consistency auxiliary head。
2. 在 temporal fusion 里使用 Doppler-aware weighting。
3. 将 static / dynamic feature aggregation 分成两条路径。
4. 在多帧累积时，用 Doppler 过滤掉不可靠的历史点。

最直接的论文参考：

- **DoppDrive**：检测前的时序对齐与融合
- **RaCFormer**：动态感知的 BEV 建模
- **CenterPoint Transformer**：辅助 velocity prediction

### 优先级 5：如果你目标是最高 leaderboard 分数，就加入 camera-radar fusion

如果目标是榜单性能，而不是最纯粹的 radar-only 叙事，这可能是上限最高的方向。

原因：

- 课程允许 camera
- 近年 VoD 上最强的方法基本都在做 radar-camera fusion
- RCBEVDet、RaCFormer、CVFusion 都证明了这一点

建议的融合顺序：

1. **Late BEV fusion baseline**
   - 图像 encoder
   - image-to-BEV lift / splat
   - 在 head 之前与 radar BEV 做融合
2. **Radar-guided depth**
   - 用 radar 改善 image-to-BEV lifting
3. **Query-based / two-stage refinement**
   - 走 RaCFormer / CVFusion 的路线

我的判断：

- 如果时间紧，不要一上来就做完整 query-based fusion
- 应该先做一个简单、稳定的 BEV fusion baseline，再逐步增强

### 优先级 6：蒸馏和预训练是强方向，但有条件

这两类方法有潜力，但要看你的约束。

#### 6.1 LiDAR teacher distillation

证据来源：

- RadarDistill
- RCTDistill（ICCV 2025，用于 radar-camera + temporal fusion）

只建议在以下前提下使用：

- 课程规则明确允许 LiDAR 只作为**训练阶段 teacher**

如果这一点不明确，就不要碰。

#### 6.2 Self-supervised radar pretraining

证据来源：

- Bootstrapping Autonomous Driving Radars with Self-Supervised Learning

只建议在以下前提下使用：

- 你有合适的原始 radar 表示，或者有大量未标注 radar 数据

对这个仓库来说，它的优先级明显低于 temporal aggregation 和 camera fusion。

## 5. 针对这个仓库的实验路线

### Phase A：低成本、高信号实验

目标：在不大改工程结构的前提下先把 baseline 提起来。

1. 加几何增强
2. 加 scheduler
3. 调 voxel size、max voxels 和 post-processing 参数
4. 做 class rebalance

为什么先做这个：

- 工程成本低
- 可以先建立更强 baseline
- 后面做结构消融时更公平

### Phase B：强化 radar-only 路线

目标：在保持仓库结构相近的前提下，做出真正有说服力的 radar-specific 提升。

1. 多帧加载
2. Ego-motion 对齐
3. 加 `dt` 通道
4. Doppler-aware aggregation
5. BEV temporal fusion block

如果你希望最后写一篇 radar-only 方向、叙事清晰的 report，这条路线是最适合的。

### Phase C：追求最高性能上限

目标：冲 leaderboard。

1. 加 monocular camera 分支
2. 先做一个 BEV fusion baseline
3. 再加 radar-guided depth
4. 最后加 proposal / query refinement

这条路线工作量更大，但和 2024-2025 在 VoD 上最强的方法最一致。

## 6. 对应的代码落点

如果我要在这个仓库里开始实现上述改动，我会优先改这些地方：

- `src/dataset/view_of_delft.py`
  - 加多帧加载
  - 加 timestamp / sweep index
  - 加 ego-motion compensation
  - 如果要做 fusion，可以顺带暴露 image path 或 camera tensor

- `src/config/model/centerpoint_radar.yaml`
  - 增加 temporal 相关配置
  - 调整更细的 voxel size
  - 增加更大 encoder / backbone 的选项
  - 增加可选 velocity head 配置

- `src/model/voxel_encoders/pillar_encoder.py`
  - RCS-aware weighting
  - Doppler-aware feature branch
  - 更强的 local attention / 更深 PFN

- `src/model/detector/centerpoint.py`
  - temporal aggregation
  - BEV recurrent block
  - 可选 camera branch

- `src/model/heads/centerpoint_head.py`
  - velocity / motion auxiliary head
  - query refinement 或 second-stage refinement

- `src/tools/train.py`
  - scheduler
  - EMA
  - 更强的 ablation logging
  - 如果在 dataloader 侧实现采样策略，这里也要配合调整

## 7. 我对下一步实验的排序建议

如果你想追求最高投入产出比，我建议按这个顺序做：

1. **先补齐训练策略**：augmentation + scheduler + threshold tuning
2. **加入 3-sweep 或 5-sweep radar accumulation**
3. **让 accumulation 变成 Doppler-aware**
4. **加入轻量级 temporal BEV module**
5. **把 encoder 升级成带 transformer / deformable block 的版本**
6. **最后再决定是坚持 radar-only 论文路线，还是转向 camera-radar fusion**

如果你的目标是尽可能追求最高性能，而不是最清晰的研究故事，那么我建议：

1. 先把 baseline 做强
2. 加 temporal radar
3. 加 camera-radar BEV fusion
4. 再加 query / proposal refinement

## 8. 总结结论

当前仓库是一个很好的**教学型 baseline**，但如果以 2025 年的 radar 检测标准来看，它明显偏弱。

最大的三个缺口是：

1. **没有时序建模**
2. **没有显式的 Doppler-aware reasoning**
3. **没有 camera fusion**

对你这份代码来说，最合理的下一步不是一开始就全面重写，而是：

1. 先补训练策略
2. 加多帧 radar aggregation，并结合 ego-motion 和 Doppler 做处理
3. 再加一个轻量级 temporal BEV fusion block

做完这一步之后，再决定要走哪条路：

- 做一个叙事清晰、实验扎实的 radar-only 方法
- 或者直接上 camera-radar fusion，冲更高 leaderboard 分数

## 9. 参考文献

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
