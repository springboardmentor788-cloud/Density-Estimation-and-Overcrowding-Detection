
import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M',
                              512, 512, 512]
        
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.frontend.load_state_dict(vgg.features[:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, dilation=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if dilation:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=2, dilation=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)