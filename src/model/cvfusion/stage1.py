from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .utils import build_bev_reference_points, project_points_to_image

try:
    from torchvision import models
except Exception:
    models = None


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CVFusionImageBackbone(nn.Module):
    def __init__(
        self,
        name='resnet18',
        pretrained=False,
        out_channels=96,
        feature_levels=('layer2', 'layer3', 'layer4'),
        freeze_bn=False,
        freeze_backbone=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.feature_levels = tuple(feature_levels)
        self.freeze_bn = freeze_bn
        self.freeze_backbone = freeze_backbone
        self.register_buffer(
            'pixel_mean',
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if models is None:
            self._build_fallback_backbone()
            return

        if name not in {'resnet18', 'resnet34'}:
            raise ValueError(f'Unsupported image backbone: {name}')

        if name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        level_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        self.projections = nn.ModuleDict({
            level: nn.Conv2d(level_channels[level], out_channels, kernel_size=1)
            for level in self.feature_levels
        })

        if self.freeze_backbone:
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in module.parameters():
                    param.requires_grad = False
        if self.freeze_bn:
            self._freeze_batch_norm()

    def _build_fallback_backbone(self):
        channels = [32, 64, 128, 256]
        stages = []
        in_channels = 3
        for out_channels in channels:
            stages.append(
                nn.Sequential(
                    ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
            )
            in_channels = out_channels
        self.fallback_stages = nn.ModuleList(stages)
        fallback_channels = {'layer1': 32, 'layer2': 64, 'layer3': 128, 'layer4': 256}
        self.projections = nn.ModuleDict({
            level: nn.Conv2d(fallback_channels[level], self.out_channels, kernel_size=1)
            for level in self.feature_levels
        })

    def _freeze_batch_norm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_batch_norm()
        return self

    def _forward_fallback(self, images):
        feats = {}
        x = images
        for stage_name, stage in zip(('layer1', 'layer2', 'layer3', 'layer4'), self.fallback_stages):
            x = stage(x)
            feats[stage_name] = x
        return [self.projections[level](feats[level]) for level in self.feature_levels]

    def forward(self, images):
        images = (images - self.pixel_mean) / self.pixel_std
        if models is None:
            return self._forward_fallback(images)

        x = self.stem(images)
        feats = {}
        feats['layer1'] = self.layer1(x)
        feats['layer2'] = self.layer2(feats['layer1'])
        feats['layer3'] = self.layer3(feats['layer2'])
        feats['layer4'] = self.layer4(feats['layer3'])
        return [self.projections[level](feats[level]) for level in self.feature_levels]


class CrossViewBEVLifter(nn.Module):
    def __init__(
        self,
        point_cloud_range,
        feat_channels=96,
        sample_heights=(0.0, 1.0, 2.0),
        min_depth=1e-3,
        score_hidden_dim=64,
    ):
        super().__init__()
        self.point_cloud_range = list(point_cloud_range)
        self.sample_heights = list(sample_heights)
        self.min_depth = float(min_depth)
        self.feat_channels = int(feat_channels)
        self.register_parameter(
            'height_embedding',
            nn.Parameter(torch.zeros(len(self.sample_heights), self.feat_channels)),
        )
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, score_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(score_hidden_dim, self.feat_channels),
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.feat_channels, score_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(score_hidden_dim, 1),
        )
        self.post_fusion = nn.Sequential(
            ConvBNReLU(self.feat_channels, self.feat_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(self.feat_channels, self.feat_channels, kernel_size=3, stride=1, padding=1),
        )

    def _sample_single_feature(self, image_feat, meta, target_shape):
        height, width = target_shape
        device = image_feat.device
        dtype = image_feat.dtype
        reference_points = build_bev_reference_points(
            target_shape,
            self.point_cloud_range,
            self.sample_heights,
            device,
            dtype,
        )
        grid, valid, depth = project_points_to_image(reference_points, meta, return_depth=True)
        sampled = F.grid_sample(
            image_feat.unsqueeze(0),
            grid.reshape(1, -1, 1, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        sampled[~valid] = 0

        num_cells = height * width
        num_heights = len(self.sample_heights)
        sampled = sampled.reshape(num_heights, num_cells, -1).permute(1, 0, 2)
        valid = valid.reshape(num_heights, num_cells).permute(1, 0).unsqueeze(-1)
        depth = torch.log1p(depth.clamp(min=self.min_depth))
        depth = depth.reshape(num_heights, num_cells, 1).permute(1, 0, 2)
        depth_feat = self.depth_encoder(depth)

        tokens = sampled + depth_feat + self.height_embedding.unsqueeze(0).to(device=device, dtype=dtype)
        logits = self.score_head(tokens).squeeze(-1)
        logits = logits.masked_fill(~valid.squeeze(-1), float('-inf'))
        has_valid = valid.any(dim=1)
        logits = torch.where(has_valid, logits, logits.new_zeros(logits.shape))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(has_valid, weights, weights.new_zeros(weights.shape))
        fused = (sampled * weights.unsqueeze(-1)).sum(dim=1)
        fused = fused.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
        return self.post_fusion(fused).squeeze(0)

    def forward(self, image_feats, metas, target_shapes: Iterable):
        outputs = []
        for image_feat, target_shape in zip(image_feats, target_shapes):
            scale_feats = [
                self._sample_single_feature(image_feat[batch_idx], metas[batch_idx], target_shape)
                for batch_idx in range(image_feat.size(0))
            ]
            outputs.append(torch.stack(scale_feats, dim=0))
        return outputs


class RGIterBEVFusion(nn.Module):
    def __init__(self, radar_channels, camera_channels):
        super().__init__()
        self.radar_channels = list(radar_channels)
        self.camera_adapters = nn.ModuleList([
            nn.Conv2d(camera_channels, channels, kernel_size=1)
            for channels in self.radar_channels
        ])
        self.occupancy_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(channels * 2, channels, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(channels, 1, kernel_size=1),
            )
            for channels in self.radar_channels
        ])
        self.cross_gate_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(channels * 2, channels, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            for channels in self.radar_channels
        ])
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(channels * 3, channels, kernel_size=3, stride=1, padding=1),
                ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1),
            )
            for channels in self.radar_channels
        ])
        self.state_blocks = nn.ModuleList([
            ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1)
            for channels in self.radar_channels
        ])
        self.prev_state_projs = nn.ModuleList([
            nn.Conv2d(self.radar_channels[idx], self.radar_channels[idx + 1], kernel_size=1)
            for idx in range(len(self.radar_channels) - 1)
        ])

    def forward(self, radar_feats, camera_bev_feats):
        fused_feats = []
        occupancy_maps = []
        propagated_state = None
        for idx, (radar_feat, camera_feat) in enumerate(zip(radar_feats, camera_bev_feats)):
            camera_feat = self.camera_adapters[idx](camera_feat)
            if propagated_state is not None:
                propagated = self.prev_state_projs[idx - 1](propagated_state)
                propagated = F.interpolate(
                    propagated,
                    size=camera_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
                camera_feat = camera_feat + propagated

            occupancy_input = torch.cat([radar_feat, camera_feat], dim=1)
            occupancy = torch.sigmoid(self.occupancy_heads[idx](occupancy_input))
            gated_camera = camera_feat * occupancy
            cross_gate = torch.sigmoid(self.cross_gate_heads[idx](occupancy_input))
            cross_response = radar_feat * cross_gate
            fusion_input = torch.cat([radar_feat, gated_camera, cross_response], dim=1)
            fused = radar_feat + self.fusion_blocks[idx](fusion_input)
            fused = fused + self.state_blocks[idx](fused * occupancy)
            propagated_state = fused
            fused_feats.append(fused)
            occupancy_maps.append(occupancy)
        return tuple(fused_feats), occupancy_maps
