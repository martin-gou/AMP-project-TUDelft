from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _as_hw(shape: Optional[Sequence[int]]):
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError(f"image_target_shape must contain [height, width], got {shape}")
    return int(shape[0]), int(shape[1])


def prepare_image_tensor(image: np.ndarray, image_target_shape=None):
    image_target_shape = _as_hw(image_target_shape)
    image = np.asarray(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    pil_image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    if image_target_shape is not None:
        target_h, target_w = image_target_shape
        pil_image = pil_image.resize((target_w, target_h), resample=Image.BILINEAR)

    image_tensor = torch.from_numpy(np.asarray(pil_image, dtype=np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return image_tensor


def project_lidar_points_to_image(
    lidar_points,
    t_camera_lidar,
    camera_projection_matrix,
    image_shape,
    image_target_shape=None,
):
    image_target_shape = _as_hw(image_target_shape)
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    if lidar_points.size == 0:
        return {
            "coords": torch.zeros((0, 2), dtype=torch.float32),
            "mask": torch.zeros((0,), dtype=torch.bool),
            "image_shape": torch.tensor(image_target_shape or image_shape[:2], dtype=torch.int64),
        }

    point_homo = np.ones((lidar_points.shape[0], 4), dtype=np.float32)
    point_homo[:, :3] = lidar_points[:, :3]

    camera_points = t_camera_lidar.dot(point_homo.T).T
    depth = camera_points[:, 2]

    uvw = camera_projection_matrix.dot(camera_points.T).T
    valid_depth = depth > 1e-5
    uv = np.zeros((lidar_points.shape[0], 2), dtype=np.float32)
    uv[valid_depth] = uvw[valid_depth, :2] / uvw[valid_depth, 2:3]

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    if image_target_shape is None:
        target_h, target_w = image_h, image_w
    else:
        target_h, target_w = image_target_shape
        uv[:, 0] *= target_w / float(image_w)
        uv[:, 1] *= target_h / float(image_h)

    mask = valid_depth.copy()
    mask &= uv[:, 0] >= 0
    mask &= uv[:, 0] <= (target_w - 1)
    mask &= uv[:, 1] >= 0
    mask &= uv[:, 1] <= (target_h - 1)

    return {
        "coords": torch.from_numpy(uv.astype(np.float32)),
        "mask": torch.from_numpy(mask.astype(np.bool_)),
        "image_shape": torch.tensor([target_h, target_w], dtype=torch.int64),
    }


def collate_vod_batch(batch):
    pts_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    image_list = []
    point_projection_list = []
    for sample in batch:
        pts_list.append(sample["lidar_data"])
        gt_labels_3d_list.append(sample["gt_labels_3d"])
        gt_bboxes_3d_list.append(sample["gt_bboxes_3d"])
        meta_list.append(sample["meta"])
        image_list.append(sample.get("image"))
        point_projection_list.append(sample.get("point_projection"))
    return dict(
        pts=pts_list,
        gt_labels_3d=gt_labels_3d_list,
        gt_bboxes_3d=gt_bboxes_3d_list,
        metas=meta_list,
        images=image_list,
        point_projections=point_projection_list,
    )
