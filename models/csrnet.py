import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):

    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(pretrained=True)

        self.frontend = list(vgg.features.children())[:23]
        self.frontend = nn.Sequential(*self.frontend)

        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(512,256,3,padding=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(256,128,3,padding=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(128,64,3,padding=2,dilation=2),
            nn.ReLU()
        )

        self.output_layer = nn.Conv2d(64,1,1)

    def forward(self,x):

        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        return x