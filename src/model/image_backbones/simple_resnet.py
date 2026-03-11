import torch
from torch import nn

try:
    from torchvision import models
except Exception:
    models = None


class ImageBackbone(nn.Module):
    def __init__(
        self,
        name='resnet18',
        pretrained=False,
        out_channels=64,
        feature_level='layer2',
    ):
        super().__init__()
        self.out_channels = out_channels
        self.feature_level = feature_level

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
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.proj = nn.Identity()
            return

        if name not in {'resnet18', 'resnet34'}:
            raise ValueError(f'Unsupported image backbone: {name}')

        if name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)

        feature_layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        ]
        feature_channels = 64

        if feature_level in {'layer2', 'layer3', 'layer4'}:
            feature_layers.append(backbone.layer2)
            feature_channels = 128
        if feature_level in {'layer3', 'layer4'}:
            feature_layers.append(backbone.layer3)
            feature_channels = 256
        if feature_level == 'layer4':
            feature_layers.append(backbone.layer4)
            feature_channels = 512

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.proj = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

    def forward(self, images):
        images = (images - self.pixel_mean) / self.pixel_std
        feats = self.feature_extractor(images)
        return self.proj(feats)
