import math

import torch
from torch import Tensor

from src.ops import boxes_iou_bev
from src.model.utils.box3d_utils import limit_period, xywhr2xyxyr
from src.model.utils.lidar_box3d import LiDARInstance3DBoxes


def meta_to_tensor(value, device, dtype=torch.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def build_bev_reference_points(shape, point_cloud_range, heights, device, dtype):
    height, width = shape
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    xs = torch.linspace(
        x_min + (x_max - x_min) / (2 * width),
        x_max - (x_max - x_min) / (2 * width),
        width,
        device=device,
        dtype=dtype,
    )
    ys = torch.linspace(
        y_min + (y_max - y_min) / (2 * height),
        y_max - (y_max - y_min) / (2 * height),
        height,
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    all_points = []
    for z in heights:
        z_tensor = torch.full((base_xy.size(0), 1), float(z), device=device, dtype=dtype)
        all_points.append(torch.cat([base_xy, z_tensor], dim=-1))
    return torch.cat(all_points, dim=0)


def project_points_to_image(points: Tensor, meta: dict):
    device = points.device
    dtype = points.dtype
    point_homo = torch.cat([points, torch.ones((points.size(0), 1), device=device, dtype=dtype)], dim=1)

    t_camera_radar = meta_to_tensor(meta.get('t_camera_radar'), device, dtype=dtype)
    camera_projection = meta_to_tensor(meta.get('camera_projection'), device, dtype=dtype)
    img_shape = meta_to_tensor(meta.get('img_shape'), device, dtype=dtype)
    ori_img_shape = meta_to_tensor(meta.get('ori_img_shape'), device, dtype=dtype)
    if t_camera_radar is None or camera_projection is None or img_shape is None:
        return points.new_zeros((points.size(0), 2)), points.new_zeros(points.size(0), dtype=torch.bool)

    camera_points = point_homo @ t_camera_radar.t()
    depth = camera_points[:, 2]
    uvw = camera_points @ camera_projection.t()
    uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-5)

    img_h, img_w = img_shape.tolist()
    if ori_img_shape is not None:
        ori_h, ori_w = ori_img_shape.tolist()
        if ori_h > 0 and ori_w > 0:
            uv[:, 0] = uv[:, 0] * (img_w / ori_w)
            uv[:, 1] = uv[:, 1] * (img_h / ori_h)

    valid = depth > 1e-3
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] <= max(img_w - 1, 0)
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] <= max(img_h - 1, 0)

    grid_x = 2.0 * (uv[:, 0] / max(img_w - 1, 1.0)) - 1.0
    grid_y = 2.0 * (uv[:, 1] / max(img_h - 1, 1.0)) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1), valid


def normalize_bev_points(points_xy: Tensor, point_cloud_range):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x = 2.0 * ((points_xy[:, 0] - x_min) / max(x_max - x_min, 1e-5)) - 1.0
    y = 2.0 * ((points_xy[:, 1] - y_min) / max(y_max - y_min, 1e-5)) - 1.0
    valid = (
        (points_xy[:, 0] >= x_min)
        & (points_xy[:, 0] <= x_max)
        & (points_xy[:, 1] >= y_min)
        & (points_xy[:, 1] <= y_max)
    )
    return torch.stack([x, y], dim=-1), valid


def boxes_to_local(points: Tensor, boxes: Tensor):
    rel = points.unsqueeze(0) - boxes[:, None, :3]
    angle = -(boxes[:, 6] + math.pi / 2)
    cos_angle = torch.cos(angle).unsqueeze(1)
    sin_angle = torch.sin(angle).unsqueeze(1)
    local_x = rel[:, :, 0] * cos_angle - rel[:, :, 1] * sin_angle
    local_y = rel[:, :, 0] * sin_angle + rel[:, :, 1] * cos_angle
    local_z = rel[:, :, 2]
    return torch.stack([local_x, local_y, local_z], dim=-1)


def local_to_global(local_points: Tensor, boxes: Tensor):
    angle = boxes[:, 6] + math.pi / 2
    cos_angle = torch.cos(angle).unsqueeze(1)
    sin_angle = torch.sin(angle).unsqueeze(1)
    world_x = local_points[:, :, 0] * cos_angle - local_points[:, :, 1] * sin_angle
    world_y = local_points[:, :, 0] * sin_angle + local_points[:, :, 1] * cos_angle
    world_z = local_points[:, :, 2]
    world = torch.stack([world_x, world_y, world_z], dim=-1)
    world[:, :, :2] += boxes[:, None, :2]
    world[:, :, 2] += boxes[:, None, 2]
    return world


def points_in_boxes_mask(points: Tensor, boxes: Tensor):
    if boxes.numel() == 0 or points.numel() == 0:
        return torch.zeros((boxes.size(0), points.size(0)), device=points.device, dtype=torch.bool)
    local = boxes_to_local(points, boxes)
    half_dims = boxes[:, None, 3:5] * 0.5
    inside_xy = local[:, :, :2].abs() <= half_dims
    inside_z = (local[:, :, 2] >= 0.0) & (local[:, :, 2] <= boxes[:, None, 5])
    return inside_xy.all(dim=-1) & inside_z


def encode_box_deltas(proposals: Tensor, targets: Tensor):
    proposal_dims = proposals[:, 3:6].clamp(min=1e-3)
    center_delta = (targets[:, :3] - proposals[:, :3]) / proposal_dims
    size_delta = torch.log(targets[:, 3:6].clamp(min=1e-3) / proposal_dims)
    yaw_delta = limit_period(targets[:, 6] - proposals[:, 6], offset=0.5, period=2 * math.pi)
    return torch.cat([center_delta, size_delta, yaw_delta.unsqueeze(-1)], dim=-1)


def decode_box_deltas(proposals: Tensor, deltas: Tensor):
    proposal_dims = proposals[:, 3:6].clamp(min=1e-3)
    centers = proposals[:, :3] + deltas[:, :3] * proposal_dims
    dims = proposal_dims * torch.exp(deltas[:, 3:6].clamp(min=-5.0, max=5.0))
    yaws = proposals[:, 6] + deltas[:, 6]
    return torch.cat([centers, dims, yaws.unsqueeze(-1)], dim=-1)


def pairwise_bev_iou(boxes_a: Tensor, boxes_b: Tensor):
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.size(0), boxes_b.size(0)))
    boxes_a_3d = LiDARInstance3DBoxes(boxes_a, box_dim=7)
    boxes_b_3d = LiDARInstance3DBoxes(boxes_b, box_dim=7)
    if boxes_a.device.type == 'cuda' and boxes_b.device.type == 'cuda':
        return boxes_iou_bev(xywhr2xyxyr(boxes_a_3d.bev), xywhr2xyxyr(boxes_b_3d.bev))

    bev_a = boxes_a_3d.nearest_bev
    bev_b = boxes_b_3d.nearest_bev
    top_left = torch.maximum(bev_a[:, None, :2], bev_b[None, :, :2])
    bottom_right = torch.minimum(bev_a[:, None, 2:], bev_b[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]
    area_a = (bev_a[:, 2] - bev_a[:, 0]).clamp(min=0) * (bev_a[:, 3] - bev_a[:, 1]).clamp(min=0)
    area_b = (bev_b[:, 2] - bev_b[:, 0]).clamp(min=0) * (bev_b[:, 3] - bev_b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)

