from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]

        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        self._load_vgg()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _load_vgg(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = list(vgg.features.children())

        frontend_layers = list(self.frontend.children())

        for i in range(len(frontend_layers)):
            if isinstance(frontend_layers[i], nn.Conv2d):
                frontend_layers[i].weight.data = vgg_features[i].weight.data
                frontend_layers[i].bias.data = vgg_features[i].bias.data

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)


def make_layers(cfg, in_channels=3, dilation=False):
    layers = []
    d_rate = 2 if dilation else 1

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=d_rate, dilation=d_rate)
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)