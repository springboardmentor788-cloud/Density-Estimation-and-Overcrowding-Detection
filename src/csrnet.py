"""
CSRNet: Dilated Convolutional Neural Networks for crowd density estimation.

Front-end: truncated VGG16 (convolutional layers only, 1/8 spatial downsampling).
Back-end: dilated convolutions for large receptive field, outputs density map.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16


class CSRNet(nn.Module):
    """
    CSRNet for crowd density map estimation.
    Input: RGB image (C, H, W). Output: density map (1, H/8, W/8).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.front_end = self._make_front_end(pretrained)
        self.back_end = self._make_back_end()

    def _make_front_end(self, pretrained: bool) -> nn.Sequential:
        """VGG16 through conv4_3 (no 4th pool) -> 1/8 size, 512 channels for back-end."""
        vgg = vgg16(weights="DEFAULT" if pretrained else None)
        # Indices 0-22: through conv4_3 (512 ch); exclude index 23 (MaxPool) to keep 1/8 resolution.
        layers = list(vgg.features.children())[:23]
        return nn.Sequential(*layers)

    def _make_back_end(self) -> nn.Sequential:
        """Dilated convolutions, same spatial size, output 1 channel."""
        return nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB, normalized.
        Returns:
            (B, 1, H/8, W/8) density map (non-negative).
        """
        x = self.front_end(x)
        x = self.back_end(x)
        return x.clamp(min=0.0)


def density_to_count(density: torch.Tensor) -> torch.Tensor:
    """Sum density map over spatial dims to get count per sample. (B,1,H,W) -> (B,)."""
    return density.clamp(min=0.0).sum(dim=(1, 2, 3))


__all__ = ["CSRNet", "density_to_count"]
