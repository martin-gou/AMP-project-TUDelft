import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.utils import prepare_image_tensor, project_lidar_points_to_image
from src.model.utils import LiDARInstance3DBoxes

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation


def transform_radar_points_to_lidar(radar_points, t_lidar_radar):
    radar_points = np.asarray(radar_points, dtype=np.float32)
    if radar_points.size == 0:
        return radar_points

    lidar_points = radar_points.copy()
    radar_xyz_hom = np.ones((radar_points.shape[0], 4), dtype=np.float32)
    radar_xyz_hom[:, :3] = radar_points[:, :3]
    lidar_points[:, :3] = homogeneous_transformation(radar_xyz_hom, t_lidar_radar)[:, :3]
    return lidar_points


def transform_points_xyz(points, transform):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    point_hom = np.ones((points.shape[0], 4), dtype=np.float32)
    point_hom[:, :3] = points[:, :3]
    transformed = points.copy()
    transformed[:, :3] = homogeneous_transformation(point_hom, transform)[:, :3]
    return transformed


class ViewOfDelft(Dataset):
    CLASSES = [
        "Car",
        "Pedestrian",
        "Cyclist",
    ]

    LABEL_MAPPING = {
        "class": 0,
        "truncated": 1,
        "occluded": 2,
        "alpha": 3,
        "bbox2d": slice(4, 8),
        "bbox3d_dimensions": slice(8, 11),
        "bbox3d_location": slice(11, 14),
        "bbox3d_rotation": 14,
    }

    def __init__(
        self,
        data_root="data/view_of_delft",
        sequential_loading=False,
        radar_sweeps=1,
        split="train",
        load_image=False,
        return_point_projection=False,
        image_target_shape=None,
    ):
        super().__init__()

        self.data_root = data_root
        assert split in ["train", "val", "test"], (
            f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        )
        self.split = split
        self.radar_sweeps = max(int(radar_sweeps), 1)
        self.load_image = load_image
        self.return_point_projection = return_point_projection
        self.image_target_shape = image_target_shape
        split_file = os.path.join(data_root, "lidar", "ImageSets", f"{split}.txt")

        with open(split_file, "r") as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]

        self.vod_kitti_locations = KittiLocations(root_dir=data_root)
        if self.radar_sweeps == 3:
            self.vod_kitti_locations.radar_dir = os.path.join(
                data_root, "radar_3frames", "training", "velodyne"
            )
        elif self.radar_sweeps == 5:
            self.vod_kitti_locations.radar_dir = os.path.join(
                data_root, "radar_5frames", "training", "velodyne"
            )

    def __len__(self):
        return len(self.sample_list)

    def _load_frame_bundle(self, idx):
        num_frame = self.sample_list[idx]
        frame_data = FrameDataLoader(
            kitti_locations=self.vod_kitti_locations,
            frame_number=num_frame,
        )
        frame_transforms = FrameTransformMatrix(frame_data)
        return num_frame, frame_data, frame_transforms

    def __getitem__(self, idx):
        num_frame, vod_frame_data, local_transforms = self._load_frame_bundle(idx)

        radar_data = transform_radar_points_to_lidar(
            vod_frame_data.radar_data,
            local_transforms.t_lidar_radar,
        )

        image_tensor = None
        point_projection = None
        if self.load_image or self.return_point_projection:
            image = vod_frame_data.image
            if image is None:
                raise FileNotFoundError(f"Image not found for frame {num_frame}")
            if self.load_image:
                image_tensor = prepare_image_tensor(image, self.image_target_shape)
            if self.return_point_projection:
                point_projection = project_lidar_points_to_image(
                    radar_data,
                    local_transforms.t_camera_lidar,
                    local_transforms.camera_projection_matrix,
                    image.shape[:2],
                    self.image_target_shape,
                )

        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != "test":
            raw_labels = vod_frame_data.raw_labels
            for label in raw_labels:
                label = label.split(" ")

                if label[self.LABEL_MAPPING["class"]] in self.CLASSES:
                    gt_labels_3d_list.append(
                        int(self.CLASSES.index(label[self.LABEL_MAPPING["class"]]))
                    )

                    bbox3d_loc_camera = np.array(
                        label[self.LABEL_MAPPING["bbox3d_location"]]
                    )
                    trans_homo_cam = np.ones((1, 4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(
                        trans_homo_cam, local_transforms.t_lidar_camera
                    )

                    bbox3d_locs = np.array(bbox3d_loc_lidar[0, :3], dtype=np.float32)
                    bbox3d_dims = np.array(
                        label[self.LABEL_MAPPING["bbox3d_dimensions"]], dtype=np.float32
                    )[[2, 1, 0]]
                    bbox3d_rot = np.array(
                        [label[self.LABEL_MAPPING["bbox3d_rotation"]]], dtype=np.float32
                    )

                    gt_bboxes_3d_list.append(
                        np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0)
                    )

        radar_data = torch.tensor(radar_data, dtype=torch.float32)

        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.array([0])
            gt_bboxes_3d = np.zeros((1, 7))
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        )

        gt_labels_3d = torch.tensor(gt_labels_3d)

        return dict(
            lidar_data=radar_data,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            image=image_tensor,
            point_projection=point_projection,
            meta=dict(
                num_frame=num_frame,
                image_shape=list(image_tensor.shape[-2:]) if image_tensor is not None else None,
            ),
        )
