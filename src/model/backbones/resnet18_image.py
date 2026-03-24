import os

import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18ImageBackbone(nn.Module):
    def __init__(
        self,
        pretrained=True,
        pretrained_path=None,
        out_stage="layer4",
        out_channels=16,
        trainable_stages=None,
    ):
        super().__init__()
        self.out_stage = out_stage
        self.trainable_stages = set(trainable_stages or [])

        self.backbone = resnet18(weights=None)
        if pretrained:
            state_dict = None
            if pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")
            if state_dict is not None:
                self.backbone.load_state_dict(state_dict)
            else:
                print(
                    f"Warning: pretrained ResNet-18 weights not found at {pretrained_path}; using random init."
                )

        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        stage_channels = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }
        if out_stage not in stage_channels:
            raise ValueError(f"Unsupported out_stage: {out_stage}")
        self.proj = nn.Conv2d(stage_channels[out_stage], out_channels, kernel_size=1, bias=False)
        self._freeze_stages()

    def _freeze_stages(self):
        modules = {
            "stem": self.stem,
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }
        for name, module in modules.items():
            requires_grad = name in self.trainable_stages
            module.requires_grad_(requires_grad)
            if not requires_grad:
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.out_stage == "layer1":
            return self.proj(x)
        x = self.layer2(x)
        if self.out_stage == "layer2":
            return self.proj(x)
        x = self.layer3(x)
        if self.out_stage == "layer3":
            return self.proj(x)
        x = self.layer4(x)
        return self.proj(x)
