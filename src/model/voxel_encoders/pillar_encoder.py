import torch
from torch import nn

from .utils import PFNLayer, get_paddings_indicator


class PointFeatureChannelGate(nn.Module):
    """Lightweight SE-style gate for point features inside each pillar."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features, mask):
        valid_points = mask.sum(dim=1).clamp_min(1.0)
        pooled = (features * mask).sum(dim=1) / valid_points
        gate = self.net(pooled).unsqueeze(1)
        return features * gate


class PointFeatureReliabilityGate(nn.Module):
    """Point-wise reliability gate for decorated pillar features."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features, mask):
        weights = self.net(features) * mask
        return features * weights


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        use_feature_gate (bool, optional): Whether to apply a lightweight
            channel gate on decorated point features before PFN aggregation.
            Defaults to False.
        feature_gate_type (str, optional): Gate type to apply when
            use_feature_gate is enabled. Options are 'channel' and 'point'.
            Defaults to 'channel'.
        feature_gate_reduction (int, optional): Reduction ratio for the
            bottleneck in the channel gate. Defaults to 4.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 use_feature_gate=False,
                 feature_gate_type='channel',
                 feature_gate_reduction=4,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max',
                 legacy=True):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        self.use_feature_gate = use_feature_gate
        assert feature_gate_type in ['channel', 'point']
        self.feature_gate_type = feature_gate_type
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        if self.use_feature_gate:
            if self.feature_gate_type == 'channel':
                self.feature_gate = PointFeatureChannelGate(
                    self.in_channels,
                    reduction=feature_gate_reduction,
                )
            else:
                self.feature_gate = PointFeatureReliabilityGate(
                    self.in_channels,
                    reduction=feature_gate_reduction,
                )
        else:
            self.feature_gate = None

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        if self.feature_gate is not None:
            features = self.feature_gate(features, mask)

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)
