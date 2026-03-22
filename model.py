import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # -------------------------
        # FRONTEND (VGG16)
        # -------------------------
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        # up to conv4_3

        # -------------------------
        # BACKEND (Dilated Conv)
        # -------------------------
        self.backend = nn.Sequential(
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
        )

        # -------------------------
        # OUTPUT LAYER
        # -------------------------
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        # 🔥 VERY IMPORTANT FIX
        x = F.relu(x)

        return x