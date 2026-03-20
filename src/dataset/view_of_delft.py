import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation


def _rotation_matrix_z(angle):
    """Return 3x3 rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)


def augment_data(points, gt_bboxes, gt_labels):
    """Apply training-time augmentations to points and GT boxes.

    Augmentations (applied in order):
      1. Random horizontal flip (y-axis)
      2. Random rotation around z-axis (±π/4)
      3. Random scaling (0.95–1.05)
      4. Small Gaussian noise on point positions

    Args:
        points: (N, C) numpy array – first 3 cols are x, y, z
        gt_bboxes: (M, 7) numpy array – [x, y, z, l, w, h, yaw]
        gt_labels: (M,) numpy array
    Returns:
        points, gt_bboxes (modified in-place or copied)
    """
    points = points.copy()
    gt_bboxes = gt_bboxes.copy()

    # 1. Random horizontal flip (flip y-axis)
    if np.random.rand() < 0.5:
        points[:, 1] = -points[:, 1]
        gt_bboxes[:, 1] = -gt_bboxes[:, 1]
        gt_bboxes[:, 6] = -gt_bboxes[:, 6]  # negate yaw

    # 2. Random rotation around z-axis
    rot_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    rot_mat = _rotation_matrix_z(rot_angle)
    points[:, :3] = points[:, :3] @ rot_mat.T
    gt_bboxes[:, :3] = gt_bboxes[:, :3] @ rot_mat.T
    gt_bboxes[:, 6] += rot_angle

    # 3. Random scaling
    scale = np.random.uniform(0.95, 1.05)
    points[:, :3] *= scale
    gt_bboxes[:, :3] *= scale
    gt_bboxes[:, 3:6] *= scale  # scale dimensions

    # 4. Small Gaussian noise on point positions
    points[:, :3] += np.random.normal(0, 0.02, size=points[:, :3].shape).astype(np.float32)

    return points, gt_bboxes


class ViewOfDelft(Dataset):
    CLASSES = ['Car', 
               'Pedestrian', 
               'Cyclist',]
            #    'rider', 
            #    'unused_bicycle', 
            #    'bicycle_rack', 
            #    'human_depiction', 
            #    'moped_or_scooter', 
            #    'motor',
            #    'truck',
            #    'other_ride',
            #    'other_vehicle',
            #    'uncertain_ride'
    
    LABEL_MAPPING = {
        'class': 0, # Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        'truncated': 1, # Not used, only there to be compatible with KITTI format.
        'occluded': 2, # Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
        'alpha': 3, # Observation angle of object, ranging [-pi..pi]
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11), # 3D object dimensions: height, width, length (in meters).
        'bbox3d_location': slice(11,14), # 3D object location x,y,z in camera coordinates (in meters).
        'bbox3d_rotation': 14, # Rotation around -Z-axis in LiDAR coordinates [-pi..pi].
    }
    
    def __init__(self, 
                 data_root = 'data/view_of_delft', 
                 sequential_loading=False,
                 split = 'train'):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')

        with open(split_file, 'r') as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]
        
        self.vod_kitti_locations = KittiLocations(root_dir = data_root)
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                         frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        radar_data = vod_frame_data.radar_data

        
        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for idx, label in enumerate(raw_labels):
                label = label.split(' ')
                
                if label[self.LABEL_MAPPING['class']] in self.CLASSES: 

                    gt_labels_3d_list.append(int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))

                    bbox3d_loc_camera = np.array(label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1,4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)
                    
                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)         
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] # hwl -> lwh
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                
                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.array([0])
            gt_bboxes_3d = np.zeros((1,7))
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)

        # Apply data augmentation during training
        if self.split == 'train' and len(gt_bboxes_3d_list) > 0:
            radar_data, gt_bboxes_3d = augment_data(
                radar_data, gt_bboxes_3d, gt_labels_3d)

        radar_data = torch.as_tensor(radar_data, dtype=torch.float32)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))
        
        gt_labels_3d = torch.tensor(gt_labels_3d)
        
        return dict(
            lidar_data = radar_data,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(
                num_frame = num_frame 
            )
        )
    

        