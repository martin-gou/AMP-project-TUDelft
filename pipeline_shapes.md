# Pipeline Data Shapes and Value Meanings

This document explains the current radar CenterPoint pipeline in this repository with:

- the Python data structure used at each stage
- the tensor shape at each stage
- what every dimension means
- what every channel / value means

It is based on the current implementation in:

- `src/dataset/view_of_delft.py`
- `src/model/detector/centerpoint.py`
- `src/ops/voxelize.py`
- `src/model/voxel_encoders/pillar_encoder.py`
- `src/model/middle_encoders/pillar_scatter.py`
- `src/model/backbones/second.py`
- `src/model/necks/second_fpn.py`
- `src/model/heads/centerpoint_head.py`
- `src/model/utils/centerpoint_bbox_coders.py`

## 1. Symbols used in this file

To avoid repeating long names, this document uses:

- `B`: batch size
- `N_i`: number of raw radar points in sample `i`
- `M_i`: number of non-empty voxels / pillars in sample `i`
- `M = sum_i M_i`: total number of non-empty voxels across the batch
- `T`: max number of points kept in one voxel
- `C`: number of channels / features
- `H, W`: BEV feature map height and width

For the current radar config:

- `T = 5`
- raw input point channels `C_raw = 7`
- voxel feature channels after PFN `C_voxel = 64`
- scatter output size is `H = 160`, `W = 160`
- head output size is `80 x 80`

## 2. Raw dataset output

### 2.1 What `__getitem__` returns

For one sample, `ViewOfDelft.__getitem__()` returns:

```python
{
    "lidar_data": radar_data,
    "gt_labels_3d": gt_labels_3d,
    "gt_bboxes_3d": gt_bboxes_3d,
    "meta": {
        "num_frame": num_frame
    }
}
```

Important note:

- the key is called `lidar_data`, but the actual content is **radar** data

### 2.2 Raw radar point cloud shape

For one sample:

```text
radar_data: [N_i, 7]
```

Meaning:

- `N_i`: number of radar points in this frame
- `7`: number of features per radar point

According to the official VoD documentation, each radar point is:

```text
[x, y, z, RCS, v_r, v_r_compensated, time]
```

Meaning of the 7 values:

1. `x`: forward position in the LiDAR / ego coordinate system, in meters
2. `y`: lateral position, in meters
3. `z`: height, in meters
4. `RCS`: Radar Cross Section, a reflectivity / radar return strength related value
5. `v_r`: raw radial Doppler velocity, in m/s
6. `v_r_compensated`: ego-motion compensated radial velocity, in m/s
7. `time`: timestamp-related value for that radar measurement

Important note:

- the notebook gives a simplified radar description with 5 channels
- the actual code config uses `in_channels: 7`
- the official VoD docs state the radar point cloud files are `Nx7`
- therefore, for this repository, the correct working assumption is **raw input points are `[N_i, 7]`**

Official source:

- VoD frame information: https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/1_frame_information/1_frame_information.html

### 2.3 Ground-truth labels shape

```text
gt_labels_3d: [K_i]
```

Meaning:

- `K_i`: number of annotated objects in this sample
- each value is a class index

Class mapping in this repo:

```text
0 -> Car
1 -> Pedestrian
2 -> Cyclist
```

Special case:

- if there is no ground-truth object, the dataset currently creates a dummy label and dummy box

### 2.4 Ground-truth box shape

Before wrapping into `LiDARInstance3DBoxes`, each box is:

```text
[x, y, z, l, w, h, yaw]
```

and the tensor shape is:

```text
gt_bboxes_3d.tensor: [K_i, 7]
```

Meaning of each box value:

1. `x`: box bottom-center x position
2. `y`: box bottom-center y position
3. `z`: box bottom-center z position
4. `l`: box length along the local forward axis
5. `w`: box width along the local lateral axis
6. `h`: box height
7. `yaw`: box heading angle around the z axis

Important coordinate convention:

- boxes are stored in **LiDAR coordinates**
- the box center in this implementation is the **bottom center**
- later, for target generation, the code converts boxes to **gravity center**

## 3. DataLoader batch structure

`collate_vod_batch()` returns:

```python
{
    "pts": [tensor_0, tensor_1, ..., tensor_{B-1}],
    "gt_labels_3d": [labels_0, labels_1, ..., labels_{B-1}],
    "gt_bboxes_3d": [boxes_0, boxes_1, ..., boxes_{B-1}],
    "metas": [meta_0, meta_1, ..., meta_{B-1}]
}
```

So `batch["pts"]` is **not** a single tensor. It is a Python list of length `B`.

For example:

```text
batch["pts"][i]           -> [N_i, 7]
batch["gt_labels_3d"][i]  -> [K_i]
batch["gt_bboxes_3d"][i]  -> LiDARInstance3DBoxes with tensor [K_i, 7]
```

## 4. Voxelization

The detector calls `self.voxel_layer()` on each sample separately.

### 4.1 Input to voxelization

For one sample:

```text
points: [N_i, 7]
```

### 4.2 Voxelization config

From the radar config:

```text
voxel_size       = [0.32, 0.32, 5]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
max_num_points   = 5
max_voxels       = 8000 (train) / 20000 (eval)
```

Meaning:

- each voxel covers `0.32 m` in x
- each voxel covers `0.32 m` in y
- each voxel covers `5 m` in z
- because z-range is also exactly `5 m`, the whole height collapses to one vertical bin

### 4.3 Grid size

Grid size is:

```text
x bins = (51.2 - 0.0) / 0.32 = 160
y bins = (25.6 - (-25.6)) / 0.32 = 160
z bins = (2 - (-3)) / 5 = 1
```

So the voxel grid size is:

```text
[160, 160, 1]
```

### 4.4 Output of voxelization for one sample

The hard voxelization operator returns:

```text
voxels_i      : [M_i, 5, 7]
coors_i       : [M_i, 3]
num_points_i  : [M_i]
```

Meaning:

- `M_i`: number of non-empty voxels in sample `i`
- `5`: at most 5 points are stored per voxel
- `7`: raw point feature dimension

Meaning of each output:

#### `voxels_i: [M_i, 5, 7]`

- dimension 0: voxel index
- dimension 1: point index inside that voxel
- dimension 2: point feature index `[x, y, z, RCS, v_r, v_r_compensated, time]`

If a voxel has fewer than 5 points:

- the remaining rows are zero-padded

#### `coors_i: [M_i, 3]`

Coordinate order is:

```text
[z_index, y_index, x_index]
```

Because there is only one z bin:

- `z_index` is almost always `0`

#### `num_points_i: [M_i]`

- `num_points_i[j]` tells you how many of the 5 slots in `voxels_i[j]` are valid points

### 4.5 Batched voxelization output

After looping over all samples, the code pads batch index in front of coordinates and concatenates all samples:

```text
voxels      : [M, 5, 7]
coors       : [M, 4]
num_points  : [M]
```

where:

```text
M = M_0 + M_1 + ... + M_{B-1}
```

Meaning of `coors`:

```text
[batch_index, z_index, y_index, x_index]
```

## 5. PillarFeatureNet

The voxel encoder takes:

```text
features   = voxels      -> [M, 5, 7]
num_points -> [M]
coors      -> [M, 4]
```

### 5.1 Feature decoration

The code adds two groups of derived features:

1. cluster-center offsets: `3` values
2. voxel-center offsets: `2` values

So:

```text
raw features              = 7
+ cluster center offsets  = 3
+ voxel center offsets    = 2
--------------------------------
decorated feature dim     = 12
```

Result:

```text
decorated_features: [M, 5, 12]
```

### 5.2 Meaning of the 12 decorated channels

For each point inside each voxel:

```text
[x, y, z, RCS, v_r, v_r_compensated, time,
 dx_cluster, dy_cluster, dz_cluster,
 dx_center, dy_center]
```

Meaning:

1. `x`: raw x position
2. `y`: raw y position
3. `z`: raw z position
4. `RCS`: radar cross section
5. `v_r`: raw radial velocity
6. `v_r_compensated`: ego-motion compensated radial velocity
7. `time`: point timestamp value
8. `dx_cluster`: point x minus mean x of points in this voxel
9. `dy_cluster`: point y minus mean y of points in this voxel
10. `dz_cluster`: point z minus mean z of points in this voxel
11. `dx_center`: point x minus geometric x center of this voxel cell
12. `dy_center`: point y minus geometric y center of this voxel cell

Important note:

- there is no `dz_center` because this is a pillar-style encoder and z is collapsed to a single vertical bin

### 5.3 PFN output

The config uses:

```text
feat_channels = [64]
```

So there is one PFN layer and it outputs:

```text
[M, 1, 64]
```

Then the code does `squeeze(1)`, so the final output of `PillarFeatureNet` is:

```text
voxel_features: [M, 64]
```

Meaning:

- each non-empty voxel / pillar is now represented by one 64-dimensional learned feature vector

## 6. PointPillarsScatter

Input:

```text
voxel_features: [M, 64]
coors         : [M, 4]
```

Output:

```text
bev_feats: [B, 64, 160, 160]
```

Meaning of each dimension:

1. `B`: batch size
2. `64`: channel dimension, one learned feature vector per BEV cell
3. `160`: y dimension of the BEV grid
4. `160`: x dimension of the BEV grid

Important detail:

- the scatter code uses coordinate order `(batch, z, y, x)`
- the z dimension is not kept as a spatial axis because the grid has only one z bin
- after scatter, the representation becomes a dense 2D BEV feature map

Interpretation:

- one cell in `[160, 160]` corresponds to one pillar on the ground plane
- each cell covers `0.32 m x 0.32 m`

## 7. SECOND backbone

Input:

```text
[B, 64, 160, 160]
```

The config says:

```text
layer_strides = [2, 2, 2]
out_channels  = [64, 128, 256]
```

So the backbone outputs 3 feature levels:

### Stage 1

```text
[B, 64, 80, 80]
```

Meaning:

- channels increase / remain at 64
- spatial size is downsampled by 2

### Stage 2

```text
[B, 128, 40, 40]
```

Meaning:

- more semantic channels
- lower spatial resolution

### Stage 3

```text
[B, 256, 20, 20]
```

Meaning:

- highest-level, most semantic feature map
- lowest spatial resolution

The backbone returns:

```python
(
    feat_stage1,   # [B, 64, 80, 80]
    feat_stage2,   # [B, 128, 40, 40]
    feat_stage3    # [B, 256, 20, 20]
)
```

## 8. SECONDFPN neck

Input:

```text
[
  [B, 64, 80, 80],
  [B, 128, 40, 40],
  [B, 256, 20, 20]
]
```

The neck config says:

```text
upsample_strides = [1, 2, 4]
out_channels     = [128, 128, 128]
```

So the neck transforms each level to:

```text
level1 -> [B, 128, 80, 80]
level2 -> [B, 128, 80, 80]
level3 -> [B, 128, 80, 80]
```

Then concatenates them along channel dimension:

```text
neck_out: [B, 384, 80, 80]
```

The neck returns:

```python
[neck_out]
```

Important note:

- the return type is a Python list with one tensor inside
- this is why the head code uses `multi_apply`

## 9. CenterHead forward output

Input to the head:

```text
[ [B, 384, 80, 80] ]
```

### 9.1 Shared conv

The head first applies a shared convolution:

```text
[B, 384, 80, 80] -> [B, 64, 80, 80]
```

### 9.2 Task heads

There are 3 separate task heads:

1. Car
2. Pedestrian
3. Cyclist

Each task predicts:

- `reg`: 2 channels
- `height`: 1 channel
- `dim`: 3 channels
- `rot`: 2 channels
- `heatmap`: 1 channel

So each task outputs a dict:

```python
{
    "reg":     [B, 2, 80, 80],
    "height":  [B, 1, 80, 80],
    "dim":     [B, 3, 80, 80],
    "rot":     [B, 2, 80, 80],
    "heatmap": [B, 1, 80, 80],
}
```

### 9.3 Meaning of every predicted value

For one spatial cell `(y, x)`:

#### `heatmap[..., y, x]`

- confidence that the object center of this class lies at this BEV cell

#### `reg[:, :, y, x] = [dx, dy]`

- sub-voxel offset inside the coarse BEV cell
- because the output grid is `80 x 80`, this offset refines the center inside the cell

#### `height[..., y, x] = z`

- predicted z coordinate of the **gravity center**

#### `dim[:, :, y, x] = [l, w, h]` in log-space

- predicted box size
- when `norm_bbox=True`, the network predicts `log(l), log(w), log(h)`
- during inference the code applies `exp()`

#### `rot[:, :, y, x] = [sin(yaw), cos(yaw)]`

- orientation encoded as sine and cosine
- final yaw is reconstructed by `atan2(sin, cos)`

### 9.4 Actual Python structure returned by the head

Because the neck returns a list with one feature level and the head uses `multi_apply`, the real return value has this structure:

```python
(
    [task0_dict],   # Car
    [task1_dict],   # Pedestrian
    [task2_dict],   # Cyclist
)
```

So:

- outer tuple length = 3 tasks
- each element is a list of length 1
- that list contains the dict for the single feature level

This is why the loss code accesses:

```python
preds_dict[0]["heatmap"]
```

## 10. Training targets

The output feature map size is:

```text
grid_size[:2] / out_size_factor = [160, 160] / 2 = [80, 80]
```

So targets are built at `80 x 80`.

For each task, target generation returns:

```text
heatmap : [B, 1, 80, 80]
anno_box: [B, 500, 8]
inds    : [B, 500]
masks   : [B, 500]
```

because:

- `max_objs = 500`
- each task has only 1 class

### 10.1 `heatmap: [B, 1, 80, 80]`

Meaning:

- a Gaussian center heatmap
- the center of each ground-truth box is drawn as a Gaussian blob
- one task corresponds to one class in this repo

### 10.2 `anno_box: [B, 500, 8]`

Each row is:

```text
[dx, dy, z, log(l), log(w), log(h), sin(yaw), cos(yaw)]
```

Meaning:

1. `dx`: center x offset inside the output grid cell
2. `dy`: center y offset inside the output grid cell
3. `z`: gravity-center z
4. `log(l)`: log length
5. `log(w)`: log width
6. `log(h)`: log height
7. `sin(yaw)`: sine of heading
8. `cos(yaw)`: cosine of heading

Important note:

- during target generation the code uses `gt_bboxes_3d.gravity_center`
- therefore the regressed `z` is the **gravity-center z**, not bottom-center z

### 10.3 `inds: [B, 500]`

Meaning:

- flattened BEV indices of object centers in the `80 x 80` output grid

Formula:

```text
ind = y * 80 + x
```

### 10.4 `masks: [B, 500]`

Meaning:

- `1` means this target slot contains a valid object
- `0` means this slot is padding

## 11. Bounding box decoding

For each task, the bbox coder starts from:

```text
heatmap  : [B, 1, 80, 80]
reg      : [B, 2, 80, 80]
height   : [B, 1, 80, 80]
dim      : [B, 3, 80, 80]
rot      : [B, 2, 80, 80]
```

### 11.1 Top-K selection

The config uses:

```text
max_num = 350
```

So each task first keeps at most:

```text
350 candidates per sample
```

### 11.2 Decoded center coordinates

The output grid cell `(x_cell, y_cell)` is converted back to metric coordinates:

```text
x = (x_cell + dx) * out_size_factor * voxel_size_x + pc_range_x_min
y = (y_cell + dy) * out_size_factor * voxel_size_y + pc_range_y_min
```

Here:

- `out_size_factor = 2`
- `voxel_size_x = 0.32`
- `voxel_size_y = 0.32`

So one output cell corresponds to:

```text
2 * 0.32 = 0.64 meters
```

before sub-cell offset refinement.

### 11.3 Decoded box shape

After decoding one task produces candidate boxes of shape:

```text
[B, <=350, 7]
```

Each decoded box is:

```text
[x, y, z, l, w, h, yaw]
```

Meaning at this moment:

1. `x`: decoded center x
2. `y`: decoded center y
3. `z`: decoded **gravity-center** z
4. `l`: decoded length
5. `w`: decoded width
6. `h`: decoded height
7. `yaw`: decoded heading

### 11.4 Conversion back to bottom center

After merging tasks, the code does:

```python
bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
```

This converts:

- `z` from gravity-center z
- back to bottom-center z

So the final `LiDARInstance3DBoxes` output again follows:

```text
[x, y, z_bottom, l, w, h, yaw]
```

## 12. Final inference output

For each sample, the detector returns:

```python
{
    "bboxes_3d": LiDARInstance3DBoxes,   # tensor shape [N_det, 7]
    "scores_3d": torch.Tensor,           # [N_det]
    "labels_3d": torch.Tensor,           # [N_det]
}
```

Meaning:

- `N_det`: number of final predicted detections after filtering and NMS
- this number is variable, not fixed

### 12.1 `bboxes_3d.tensor: [N_det, 7]`

Each row is:

```text
[x, y, z_bottom, l, w, h, yaw]
```

Meaning:

1. `x`: predicted box bottom-center x in LiDAR coordinates
2. `y`: predicted box bottom-center y in LiDAR coordinates
3. `z_bottom`: predicted box bottom-center z
4. `l`: predicted length
5. `w`: predicted width
6. `h`: predicted height
7. `yaw`: predicted orientation angle

### 12.2 `scores_3d: [N_det]`

- detection confidence score

### 12.3 `labels_3d: [N_det]`

Class id:

```text
0 -> Car
1 -> Pedestrian
2 -> Cyclist
```

## 13. Full shape summary

```text
Dataset sample:
  radar_data                [N_i, 7]
  gt_labels_3d              [K_i]
  gt_bboxes_3d.tensor       [K_i, 7]

Batch:
  pts                       list of B tensors
  pts[i]                    [N_i, 7]

Voxelization:
  voxels                    [M, 5, 7]
  coors                     [M, 4]      = [batch, z, y, x]
  num_points                [M]

PillarFeatureNet:
  decorated features        [M, 5, 12]
  voxel_features            [M, 64]

Scatter:
  bev_feats                 [B, 64, 160, 160]

SECOND backbone:
  stage1                    [B, 64, 80, 80]
  stage2                    [B, 128, 40, 40]
  stage3                    [B, 256, 20, 20]

SECONDFPN:
  neck_out                  [B, 384, 80, 80]

CenterHead per task:
  reg                       [B, 2, 80, 80]
  height                    [B, 1, 80, 80]
  dim                       [B, 3, 80, 80]
  rot                       [B, 2, 80, 80]
  heatmap                   [B, 1, 80, 80]

Training target per task:
  heatmap                   [B, 1, 80, 80]
  anno_box                  [B, 500, 8]
  inds                      [B, 500]
  masks                     [B, 500]

Final output per sample:
  bboxes_3d.tensor          [N_det, 7]
  scores_3d                 [N_det]
  labels_3d                 [N_det]
```

## 14. One easy way to verify shapes on DelftBlue

If you want to check the real runtime shapes in your environment, add temporary print statements at:

- `src/dataset/view_of_delft.py`
- `src/model/detector/centerpoint.py`
- `src/model/voxel_encoders/pillar_encoder.py`
- `src/model/middle_encoders/pillar_scatter.py`
- `src/model/backbones/second.py`
- `src/model/necks/second_fpn.py`
- `src/model/heads/centerpoint_head.py`

The most important first check is:

```python
print(radar_data.shape)
print(radar_data[:3])
```

inside `ViewOfDelft.__getitem__()`, because that confirms the exact raw point format used on your DelftBlue environment.

## 15. Reference for raw radar point format

The 7-channel radar point format used above is supported by the official View-of-Delft documentation:

- https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/1_frame_information/1_frame_information.html

It states that the radar point cloud files are stored as `Nx7` arrays with:

```text
[x, y, z, RCS, v_r, v_r_compensated, time]
```
