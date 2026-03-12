import torch
import torch.nn.functional as F
from torch import nn


class PointImageFusion(nn.Module):
    def __init__(
        self,
        image_feat_dim=64,
        fused_image_dim=16,
        point_time_index=6,
        current_flag_index=7,
        voxel_feature_dim=64,
        add_img_valid_flag=True,
        use_camera=True,
    ):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.fused_image_dim = fused_image_dim
        self.point_time_index = point_time_index
        self.current_flag_index = current_flag_index
        self.voxel_feature_dim = voxel_feature_dim
        self.add_img_valid_flag = add_img_valid_flag
        self.use_camera = use_camera

        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, fused_image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_image_dim, fused_image_dim),
        )
        fusion_in_channels = voxel_feature_dim + fused_image_dim + (1 if add_img_valid_flag else 0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_channels, voxel_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )

    def _meta_to_tensor(self, value, device, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def _sample_image_features(self, voxel_xyz, voxel_mask, img_feat, meta):
        num_voxels = voxel_xyz.size(0)
        device = voxel_xyz.device
        sampled = voxel_xyz.new_zeros((num_voxels, self.image_feat_dim))
        img_valid = voxel_xyz.new_zeros((num_voxels, 1))
        if img_feat is None or not voxel_mask.any() or meta.get('img_shape') is None:
            return sampled, img_valid

        current_indices = torch.nonzero(voxel_mask, as_tuple=False).squeeze(1)
        current_points = voxel_xyz[current_indices]
        point_homo = torch.cat(
            [current_points, current_points.new_ones((current_points.size(0), 1))],
            dim=1,
        )

        t_camera_radar = self._meta_to_tensor(meta['t_camera_radar'], device)
        camera_projection = self._meta_to_tensor(meta['camera_projection'], device)
        img_shape = self._meta_to_tensor(meta['img_shape'], device, dtype=torch.float32)
        ori_img_shape = self._meta_to_tensor(meta['ori_img_shape'], device, dtype=torch.float32)
        img_h, img_w = img_shape.tolist()
        ori_h, ori_w = ori_img_shape.tolist()

        camera_points = point_homo @ t_camera_radar.t()
        depth = camera_points[:, 2]
        uvw = camera_points @ camera_projection.t()
        uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-5)
        if ori_h > 0 and ori_w > 0:
            uv[:, 0] = uv[:, 0] * (img_w / ori_w)
            uv[:, 1] = uv[:, 1] * (img_h / ori_h)

        valid = depth > 0
        valid &= uv[:, 0] >= 0
        valid &= uv[:, 0] <= max(img_w - 1, 0)
        valid &= uv[:, 1] >= 0
        valid &= uv[:, 1] <= max(img_h - 1, 0)

        grid_x = 2.0 * (uv[:, 0] / max(img_w - 1, 1.0)) - 1.0
        grid_y = 2.0 * (uv[:, 1] / max(img_h - 1, 1.0)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

        sampled_current = F.grid_sample(
            img_feat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        sampled_current = sampled_current.squeeze(0).squeeze(-1).transpose(0, 1)
        sampled_current[~valid] = 0

        sampled[current_indices] = sampled_current
        img_valid[current_indices, 0] = valid.float()
        return sampled, img_valid

    def forward(self, voxel_features, voxels, num_points, coors, img_feats=None, metas=None):
        if img_feats is None or not self.use_camera or metas is None:
            return voxel_features

        device = voxel_features.device
        voxel_count = voxels.size(1)
        point_mask = torch.arange(voxel_count, device=device).unsqueeze(0) < num_points.unsqueeze(1)
        current_point_mask = voxels[:, :, self.current_flag_index] > 0.5
        current_point_mask &= point_mask
        current_counts = current_point_mask.sum(dim=1, keepdim=True)
        voxel_current_mask = current_counts.squeeze(1) > 0

        current_xyz = (voxels[:, :, :3] * current_point_mask.unsqueeze(-1)).sum(dim=1)
        current_xyz = current_xyz / current_counts.clamp(min=1).to(voxels.dtype)

        raw_img_features = voxel_features.new_zeros((voxel_features.size(0), self.image_feat_dim))
        img_valid = voxel_features.new_zeros((voxel_features.size(0), 1))

        for batch_idx in range(img_feats.size(0)):
            batch_mask = coors[:, 0] == batch_idx
            if not batch_mask.any():
                continue
            batch_xyz = current_xyz[batch_mask]
            batch_current_mask = voxel_current_mask[batch_mask]
            sampled_batch, valid_batch = self._sample_image_features(
                batch_xyz,
                batch_current_mask,
                img_feats[batch_idx:batch_idx + 1],
                metas[batch_idx],
            )
            raw_img_features[batch_mask] = sampled_batch
            img_valid[batch_mask] = valid_batch

        img_features = self.image_proj(raw_img_features) * img_valid
        fusion_parts = [voxel_features]
        if self.add_img_valid_flag:
            fusion_parts.append(img_valid)
        fusion_parts.append(img_features)
        fusion_input = torch.cat(fusion_parts, dim=1)
        return voxel_features + self.fusion_proj(fusion_input)
