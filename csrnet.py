import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class CSRNet(nn.Module):

    def __init__(self):
        super(CSRNet, self).__init__()

        # ✅ Pretrained VGG16 frontend
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        print("✅ Frontend is trainable")

        # ✅ Backend (dilated conv layers)
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

        # ✅ Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # ✅ Initialize weights
        self._initialize_weights()

    def forward(self, x):

        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

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
