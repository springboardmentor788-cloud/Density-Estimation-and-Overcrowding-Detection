import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):

    def __init__(self):

        super(CSRNet, self).__init__()

        frontend = list(models.vgg16(pretrained=True).features.children())[:23]

        self.frontend = nn.Sequential(*frontend)

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

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        return x