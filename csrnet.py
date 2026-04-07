import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class CSRNet(nn.Module):

    def __init__(self, freeze_frontend=True):
        super(CSRNet, self).__init__()

        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # 🔥 Freeze frontend
        if freeze_frontend:
            print("Freezing frontend...")
            for param in self.frontend.parameters():
                param.requires_grad = False

        self.backend = nn.Sequential(
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

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # 🔥 CRITICAL FIX
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):

        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        # 🔥 ensure non-negative density
        x = self.relu(x)

        return x

    def _initialize_weights(self):

        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)