import torch
from torch import nn


class TemporalVoxelAttention(nn.Module):
    def __init__(
        self,
        voxel_feature_dim=64,
        point_feature_dim=8,
        current_flag_index=7,
        num_heads=4,
        dropout=0.0,
        use_history_gate=True,
    ):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.point_feature_dim = point_feature_dim
        self.current_flag_index = current_flag_index
        self.use_history_gate = use_history_gate

        self.point_proj = nn.Sequential(
            nn.Linear(point_feature_dim, voxel_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=voxel_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(voxel_feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(voxel_feature_dim, voxel_feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(voxel_feature_dim * 2, voxel_feature_dim),
        )
        self.norm2 = nn.LayerNorm(voxel_feature_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )

    def _masked_mean(self, values, mask):
        weighted = values * mask.unsqueeze(-1).to(values.dtype)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(values.dtype)
        return weighted.sum(dim=1) / counts

    def forward(self, voxel_features, voxels, num_points):
        if voxels.size(-1) <= self.current_flag_index:
            return voxel_features

        device = voxel_features.device
        voxel_count = voxels.size(1)
        point_mask = torch.arange(voxel_count, device=device).unsqueeze(0) < num_points.unsqueeze(1)

        current_mask = (voxels[:, :, self.current_flag_index] > 0.5) & point_mask
        history_mask = point_mask & ~current_mask
        history_present = history_mask.any(dim=1, keepdim=True)

        point_features = voxels[:, :, :self.point_feature_dim]
        current_summary = self._masked_mean(point_features, current_mask)
        history_summary = self._masked_mean(point_features, history_mask)

        voxel_token = voxel_features.unsqueeze(1)
        current_token = self.point_proj(current_summary).unsqueeze(1)
        history_token = self.point_proj(history_summary).unsqueeze(1)
        tokens = torch.cat([voxel_token, current_token, history_token], dim=1)

        key_padding_mask = torch.cat(
            [
                torch.zeros((voxels.size(0), 1), dtype=torch.bool, device=device),
                ~current_mask.any(dim=1, keepdim=True),
                ~history_present,
            ],
            dim=1,
        )

        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        refined = self.output_proj(tokens[:, 0])
        if self.use_history_gate:
            refined = refined * history_present.to(refined.dtype)
        return voxel_features + refined
