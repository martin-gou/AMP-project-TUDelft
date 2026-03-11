import torch
import torch.nn.functional as F
from torch import nn


class PointImageFusion(nn.Module):
    def __init__(
        self,
        image_feat_dim=64,
        fused_image_dim=16,
        point_time_index=6,
        add_img_valid_flag=True,
        use_camera=True,
    ):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.fused_image_dim = fused_image_dim
        self.point_time_index = point_time_index
        self.add_img_valid_flag = add_img_valid_flag
        self.use_camera = use_camera

        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, fused_image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_image_dim, fused_image_dim),
        )

    def _meta_to_tensor(self, value, device, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def _current_sweep_mask(self, points):
        if points.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=points.device)
        times = points[:, self.point_time_index]
        current_time = times[torch.argmin(times.abs())]
        return (times - current_time).abs() <= 1e-4

    def _sample_image_features(self, points, current_mask, img_feat, meta):
        num_points = points.size(0)
        device = points.device
        sampled = points.new_zeros((num_points, self.image_feat_dim))
        img_valid = points.new_zeros((num_points, 1))
        if img_feat is None or not current_mask.any():
            return sampled, img_valid

        current_indices = torch.nonzero(current_mask, as_tuple=False).squeeze(1)
        current_points = points[current_indices, :3]
        point_homo = torch.cat(
            [current_points, current_points.new_ones((current_points.size(0), 1))],
            dim=1,
        )

        t_camera_radar = self._meta_to_tensor(meta['t_camera_radar'], device)
        camera_projection = self._meta_to_tensor(meta['camera_projection'], device)
        img_shape = self._meta_to_tensor(meta['img_shape'], device, dtype=torch.float32)
        img_h, img_w = img_shape.tolist()

        camera_points = point_homo @ t_camera_radar.t()
        depth = camera_points[:, 2]
        uvw = camera_points @ camera_projection.t()
        uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-5)

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

    def forward(self, pts_list, img_feats=None, metas=None):
        fused_points = []
        for batch_idx, points in enumerate(pts_list):
            device = img_feats.device if isinstance(img_feats, torch.Tensor) else points.device
            points = points.to(device)
            current_mask = self._current_sweep_mask(points)
            current_flag = current_mask.float().unsqueeze(1)

            if img_feats is not None and self.use_camera:
                raw_img_features, img_valid = self._sample_image_features(
                    points,
                    current_mask,
                    img_feats[batch_idx:batch_idx + 1],
                    metas[batch_idx],
                )
            else:
                raw_img_features = points.new_zeros((points.size(0), self.image_feat_dim))
                img_valid = points.new_zeros((points.size(0), 1))

            img_features = self.image_proj(raw_img_features) * img_valid

            parts = [points, current_flag]
            if self.add_img_valid_flag:
                parts.append(img_valid)
            parts.append(img_features)
            fused_points.append(torch.cat(parts, dim=1))

        return fused_points
