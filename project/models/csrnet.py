from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_pretrained_frontend: bool = True, use_batch_norm: bool = False) -> None:
        super().__init__()
        self.frontend_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_cfg = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_cfg, batch_norm=use_batch_norm, dilation=False)
        self.backend = make_layers(self.backend_cfg, in_channels=512, batch_norm=use_batch_norm, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if load_pretrained_frontend:
            self._load_vgg16_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _load_vgg16_frontend(self) -> None:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        frontend_state = self.frontend.state_dict()
        vgg_state = vgg.features.state_dict()

        mapped = {}
        for k in frontend_state.keys():
            if k in vgg_state and frontend_state[k].shape == vgg_state[k].shape:
                mapped[k] = vgg_state[k]

        frontend_state.update(mapped)
        self.frontend.load_state_dict(frontend_state)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(
    cfg,
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    d_rate = 2 if dilation else 1
    layers = []

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate,
            )
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v

    return nn.Sequential(*layers)
