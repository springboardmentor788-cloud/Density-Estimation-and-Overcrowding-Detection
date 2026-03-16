"""
CSRNet (Contextual Spatial Resolution Network) implementation for crowd density estimation.
Based on the paper: "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"

This model uses VGG-16 as backbone for feature extraction and dilated convolutions
to maintain large receptive fields while preserving spatial resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet implementation for crowd density estimation.
    
    Architecture:
    - VGG-16 backbone (first 10 layers) for feature extraction
    - Dilated convolutional layers for density map generation
    - Output: Single channel density map
    """
    
    def __init__(self, pretrained=True):
        """
        Initialize CSRNet model.
        
        Args:
            pretrained (bool): Whether to use pretrained VGG-16 weights
        """
        super(CSRNet, self).__init__()
        
        # Load VGG-16 backbone
        if pretrained:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg16(weights=None)
        features = list(vgg.features[:23])  # Take first 23 layers (up to pool5)
        self.frontend = nn.ModuleList(features)
        
        # Backend - dilated convolutional layers for density map generation
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # Final layer outputs single channel density map
        )
        
        # Initialize backend weights
        self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Density maps [batch_size, 1, H//8, W//8]
        """
        # Frontend feature extraction
        for layer in self.frontend:
            x = layer(x)
        
        # Backend density estimation
        x = self.backend(x)
        
        # Ensure positive values using ReLU
        x = F.relu(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize backend weights using normal distribution."""
        for module in self.backend:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_density_map(self, input_image):
        """
        Get density map prediction for a single image.
        
        Args:
            input_image (torch.Tensor): Input image [1, 3, H, W] or [3, H, W]
            
        Returns:
            torch.Tensor: Density map prediction
        """
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)
            
        with torch.no_grad():
            density_map = self.forward(input_image)
        
        return density_map.squeeze()
    
    def count_people(self, input_image):
        """
        Estimate total people count in the image.
        
        Args:
            input_image (torch.Tensor): Input image [1, 3, H, W] or [3, H, W]
            
        Returns:
            float: Estimated people count
        """
        density_map = self.get_density_map(input_image)
        return torch.sum(density_map).item()


def create_csrnet(pretrained=True):
    """
    Factory function to create CSRNet model.
    
    Args:
        pretrained (bool): Whether to use pretrained VGG-16 weights
        
    Returns:
        CSRNet: Initialized CSRNet model
    """
    model = CSRNet(pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Example usage and model testing
    print("Testing CSRNet model...")
    
    # Create model
    model = create_csrnet(pretrained=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Estimated count: {model.count_people(dummy_input):.2f}")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")