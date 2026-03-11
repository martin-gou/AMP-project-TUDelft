# Implementation Plan

日期：2026-03-12

## 1. 目标

在当前 `martin/codex_run` 实验分支上，把现有的 radar-only CenterPoint baseline 扩展为：

1. 支持 **temporal radar aggregation**
2. 支持 **camera-radar fusion**
3. 尽量复用现有 `voxelize -> PFN -> scatter -> SECOND -> FPN -> CenterHead` 主干
4. 第一版优先保证：
   - 能训练
   - 能 ablate
   - 改动边界清晰

## 2. 采用的第一版方案

第一版不做单目 BEV lifting，不做 query-based fusion，不重写 detector head。

第一版采用：

**多帧 radar 聚合 + 当前帧图像点级融合**

整体结构：

```text
current image
  -> image backbone
  -> image feature map

current radar + past K-1 radar sweeps
  -> ego-motion compensation to current frame
  -> concatenate sweeps
  -> add temporal channels

current-sweep radar points
  -> project to image
  -> sample image features
  -> fuse image feature into point feature

fused radar points
  -> voxelization
  -> pillar encoder
  -> scatter
  -> SECOND
  -> SECONDFPN
  -> CenterHead
```

## 3. 为什么选这个方案

这个方案是当前仓库里最合适的第一版，因为它：

1. 最大程度复用现有 detector
2. 不需要先做单目 depth / image-to-BEV
3. camera 和 temporal 两部分都能独立做 ablation
4. 风险主要集中在 dataset 和 fusion，不会把整个训练流程打碎

## 4. 第一版的关键设计决策

### 4.1 Temporal radar

第一版做 `K=3` sweeps：

- 当前帧 `t0`
- 历史一帧 `t-1`
- 历史两帧 `t-2`

每个 radar point 除原始 7 维外，再增加：

1. `dt`
2. `is_current_sweep`

如果后续验证稳定，可再增加：

3. `sweep_idx`

第一版建议保留最小 temporal 扩展：

```text
[x, y, z, RCS, v_r, v_r_comp, time, dt, is_current]
```

### 4.2 Camera fusion

第一版只用 **当前帧图像**。

只给 **当前 sweep radar points** 采样 image feature。

历史 sweeps 的点：

- image feature 置零
- 增加一个 `img_valid=0`

当前 sweep 的点：

- 若能投影到图像内并成功采样，则 `img_valid=1`
- 否则 `img_valid=0`

所以 point-level fused feature 推荐形式为：

```text
[x, y, z, RCS, v_r, v_r_comp, time, dt, is_current, img_valid, img_f1, ..., img_fk]
```

### 4.3 图像特征维度

不要直接拼接高维 feature。

第一版：

- image backbone 输出中间特征
- 对每个 radar point 采样后
- 用一个小的 MLP / Linear 压到 `k=16`

建议第一版最终 point feature 维度：

```text
7 radar
+ 1 dt
+ 1 is_current
+ 1 img_valid
+ 16 image feature
= 26 dims
```

因此配置里：

```yaml
voxel_encoder.in_channels: 26
```

### 4.4 不在第一版做的内容

第一版不做：

1. monocular BEV lifting
2. query-based fusion
3. second-stage refinement
4. velocity head
5. temporal BEV recurrent block
6. Doppler-aware learned weighting

这些都可以作为第二阶段增强。

## 5. 代码改动计划

## 5.1 Dataset 层

文件：

- `src/dataset/view_of_delft.py`
- `src/dataset/utils.py`

需要实现的内容：

### A. 当前帧图像加载

`ViewOfDelft.__getitem__()` 新增返回：

- `image`
- `img_shape`
- 相机投影相关矩阵

建议返回结构：

```python
return dict(
    lidar_data=fused_or_temporal_radar_points,
    image=image_tensor,
    gt_labels_3d=gt_labels_3d,
    gt_bboxes_3d=gt_bboxes_3d,
    meta=dict(
        num_frame=num_frame,
        img_shape=(H, W),
        lidar_to_camera=...,
        camera_to_lidar=...,
        camera_projection=...,
    )
)
```

### B. 时序 radar 读取

需要支持读取历史 `K-1` 帧。

实现方式：

1. 根据当前 frame id 找到同 sequence 内前几帧
2. 读取各帧 `radar_data`
3. 读取每帧 pose / transform
4. 将历史帧点云变换到当前帧 LiDAR 坐标系

### C. temporal feature augmentation

对每个 radar point 添加：

- `dt`
- `is_current_sweep`

历史帧点的 `dt < 0`，当前帧 `dt = 0`

### D. collate function 扩展

`collate_vod_batch()` 需要新增：

- `imgs`
- `metas` 中更完整的 calibration 信息

返回结构变为：

```python
dict(
    pts=pts_list,
    imgs=img_list,
    gt_labels_3d=gt_labels_3d_list,
    gt_bboxes_3d=gt_bboxes_3d_list,
    metas=meta_list,
)
```

## 5.2 新增 image backbone

建议新增目录：

- `src/model/image_backbones/`

建议新增文件：

- `src/model/image_backbones/__init__.py`
- `src/model/image_backbones/simple_resnet.py`

第一版目标：

- 使用轻量 backbone
- 输出单层或双层 feature map

推荐方案：

- `torchvision` 预训练 `resnet18` 或 `resnet34`
- 取 `C3` 或 `C4` feature
- 接一个 `1x1 conv` 压到统一通道数，例如 `64`

第一版优先：

- 单层 feature map
- 实现简单

## 5.3 新增 point fusion 模块

建议新增目录：

- `src/model/fusion/`

建议新增文件：

- `src/model/fusion/__init__.py`
- `src/model/fusion/point_image_fusion.py`

模块职责：

输入：

- `pts_list`
- `img_feats`
- `metas`

输出：

- 带 image feature 的 point list

主要步骤：

1. 提取当前 sweep points
2. 用标定矩阵投影到图像
3. 归一化到 feature map 坐标
4. 用 `grid_sample` 做双线性采样
5. 采样结果经过小 MLP / Linear 降维
6. 拼接回原始 point feature
7. 对历史点填零 image feature，并设置 `img_valid=0`

可选增强：

- 再加一个轻量 gate：
  - 输入 radar feature + sampled image feature
  - 输出一个 `[0,1]` 权重
  - 用于调制 image feature 注入强度

第一版若时间紧，可先不做 gate。

## 5.4 修改 detector

文件：

- `src/model/detector/centerpoint.py`

需要改动：

### A. 构造函数新增组件

新增：

- `self.use_camera`
- `self.num_sweeps`
- `self.img_backbone`
- `self.point_fusion`

### B. `_model_forward` 扩展输入

现状：

```python
def _model_forward(self, pts_data):
```

修改为：

```python
def _model_forward(self, pts_data, imgs=None, metas=None):
```

流程：

1. 若 `use_camera`：
   - `img_feats = self.img_backbone(imgs)`
   - `pts_data = self.point_fusion(pts_data, img_feats, metas)`
2. 继续原始 radar 流程：
   - voxelize
   - voxel_encoder
   - scatter
   - backbone
   - neck
   - head

### C. training / validation / test step 改为同时传入 `imgs`

例如：

```python
imgs = batch.get("imgs", None)
ret_dict = self._model_forward(pts_data, imgs=imgs, metas=metas)
```

## 5.5 配置层

新增 config 文件：

- `src/config/model/centerpoint_radar_camera_temporal.yaml`

不要直接覆盖原 `centerpoint_radar.yaml`，保留 baseline 可重复性。

建议新增字段：

```yaml
name: centerpoint_radar_camera_temporal

use_camera: true
num_sweeps: 3

image_backbone:
  name: resnet18
  pretrained: true
  out_channels: 64

fusion:
  type: point_image_fusion
  image_feat_dim: 64
  fused_image_dim: 16
  add_img_valid_flag: true

voxel_encoder:
  in_channels: 26
  feat_channels: [64]
```

同时在 train / eval / test config 里保留兼容：

- baseline model
- new model

## 6. 实现顺序

我建议按以下顺序做 commit。

### Commit 1: Dataset plumbing

目标：

- 只把 image 和 calibration 返回出来
- 暂时不做 temporal

验收标准：

- dataloader 能跑
- batch 里能拿到 `imgs`
- batch 里能拿到投影矩阵

### Commit 2: Temporal radar aggregation

目标：

- 加 `num_sweeps=3`
- 历史 radar points 对齐到当前帧
- 加 `dt` 和 `is_current`

验收标准：

- `pts[i]` shape 从 `[N_i, 7]` 变成 `[N_i_total, 9]`
- 训练流程仍然能跑到 voxelization 前

### Commit 3: Image backbone

目标：

- 接入 image backbone
- forward 能输出 image feature map

验收标准：

- `imgs -> img_feats` shape 正常
- 不影响原 radar-only 分支

### Commit 4: Point-image fusion

目标：

- 当前 sweep 点投影到 image feature
- 采样 image feature 并拼回 point feature

验收标准：

- fused point shape 变成 `[N_i_total, 26]`
- `voxel_encoder.in_channels` 对齐
- 整个 forward 能跑通

### Commit 5: End-to-end training

目标：

- 跑通训练
- 跑通验证
- 不崩溃

验收标准：

- loss 正常下降
- validation 有有效框输出

### Commit 6: Basic ablation support

增加可开关项：

- `use_camera`
- `num_sweeps`
- `zero_image_feature_for_history`

这样可以跑三组基线：

1. radar-only single frame
2. radar-only temporal
3. radar + temporal + camera

## 7. 推荐实验顺序

先不要一口气把所有东西一起训练。

建议实验顺序：

1. baseline
   - 原始 radar-only 单帧
2. temporal-only
   - 3-sweep radar
3. camera-only
   - 单帧 radar + current image point fusion
4. temporal + camera
   - 3-sweep radar + current image point fusion

这样你能知道提升来自哪里。

## 8. 训练策略同步改动

在多模态和时序加入后，我建议同时补以下训练细节：

1. scheduler
   - cosine decay 或 one-cycle
2. augmentation
   - BEV flip / rotation / scaling
3. threshold tuning
   - `score_threshold`
   - `nms_thr`
   - `post_max_size`

这些可以不和结构改动绑在一个 commit 里，但建议在第一轮跑通后尽快加上。

## 9. 风险点

### 风险 1：VoD API 的时序 / pose 访问方式

这是最大的技术风险。

需要先确认：

- 如何找到历史帧
- 如何拿到每帧 pose
- 如何把历史 radar 变换到当前帧

如果 API 不能直接给，需要自己解析对应文件。

### 风险 2：图像和 radar 时间不同步

第一版可以接受轻微不同步，因为我们只给当前 sweep 做 image sampling。

### 风险 3：历史动态点和当前图像错配

这就是为什么第一版只给当前 sweep 加 image feature。

### 风险 4：point feature 维度变大后 PFN 不稳定

缓解方式：

- image feature 压到 8 或 16 维
- 先不用太深的 image backbone

## 10. 第二阶段增强

第一版跑通后，优先考虑这些增强：

1. Doppler-aware temporal weighting
2. temporal BEV fusion block
3. multi-scale image feature sampling
4. point-image gating
5. EMA

其中我最看好的顺序是：

1. Doppler-aware temporal weighting
2. point-image gating
3. temporal BEV block

## 11. 最终希望达到的 ablation 表

最终 report 至少应有以下实验：

1. Baseline radar-only single frame
2. + Temporal radar
3. + Camera point fusion
4. + Temporal radar + camera point fusion
5. + Training recipe improvements

这样你的故事线会非常清楚：

- baseline 为什么弱
- temporal 为什么值
- camera 为什么值
- 两者是否互补

## 12. 下一步执行建议

下一步我建议立刻开始：

1. 先确认 VoD dataset API 中：
   - image path / image array
   - projection matrix
   - 历史帧访问方式
   - ego pose / transform
2. 然后做 `Commit 1 + Commit 2`

也就是先把：

- image loading
- calibration loading
- temporal radar aggregation

打通，再开始写 fusion module。
