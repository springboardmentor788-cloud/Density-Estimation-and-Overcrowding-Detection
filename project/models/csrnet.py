from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, *, kernel_size: int = 3, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
        nn.ReLU(inplace=True),
    )


def _make_frontend() -> nn.Sequential:
    layers = [
        _conv_block(3, 64),
        _conv_block(64, 64),
        nn.MaxPool2d(2),
        _conv_block(64, 128),
        _conv_block(128, 128),
        nn.MaxPool2d(2),
        _conv_block(128, 256),
        _conv_block(256, 256),
        _conv_block(256, 256),
        nn.MaxPool2d(2),
        _conv_block(256, 512),
        _conv_block(512, 512),
        _conv_block(512, 512),
    ]
    return nn.Sequential(*layers)


def _make_backend() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 1, kernel_size=1),
    )


class CSRNet(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str | Path | None = None) -> None:
        super().__init__()
        self.frontend = _make_frontend()
        self.backend = _make_backend()

        if pretrained_path is not None:
            self.load_weights(pretrained_path)
        elif pretrained:
            self._try_load_torchvision_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return torch.relu(x)

    def _try_load_torchvision_frontend(self) -> None:
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception:
            return

        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            vgg_features = list(vgg.features.children())[:23]
            source = nn.Sequential(*vgg_features)
            self.frontend.load_state_dict(source.state_dict(), strict=False)
        except Exception:
            return

    def load_weights(self, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(str(checkpoint_path), map_location=map_location)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"], strict=False)
        elif isinstance(state, dict):
            self.load_state_dict(state, strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        density = self(x)
        return density.flatten(1).sum(dim=1)
