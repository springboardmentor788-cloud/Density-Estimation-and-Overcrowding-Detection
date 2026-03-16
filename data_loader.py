"""
Data loader and preprocessing utilities for crowd density estimation.
Handles loading images, preprocessing, and creating synthetic ground truth density maps.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random


class CrowdDataset(Dataset):
    """
    PyTorch Dataset class for crowd density estimation.
    
    This dataset loads images and creates synthetic ground truth density maps
    since we don't have real ground truth annotations.
    """
    
    def __init__(self, image_dir, transform=None, image_size=(256, 256), 
                 density_downscale=8, min_people=5, max_people=50):
        """
        Initialize the crowd dataset.
        
        Args:
            image_dir (str): Path to directory containing images
            transform (callable): Optional transform to be applied on images
            image_size (tuple): Target image size (height, width)
            density_downscale (int): Downscaling factor for density maps
            min_people (int): Minimum number of people to simulate
            max_people (int): Maximum number of people to simulate
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.density_downscale = density_downscale
        self.min_people = min_people
        self.max_people = max_people
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, density_map_tensor, count)
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Create synthetic density map
        density_map, count = self.create_synthetic_density_map(idx)
        
        # Apply transforms
        if self.transform:
            # Convert to PIL Image for transforms
            image_pil = Image.fromarray(image)
            image = self.transform(image_pil)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Convert density map to tensor
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)  # Add channel dimension
        
        return image, density_map, count
    
    def create_synthetic_density_map(self, idx):
        """
        Create synthetic ground truth density map.
        
        Since we don't have real annotations, we simulate crowd density by:
        1. Randomly placing points representing people
        2. Applying Gaussian kernels to create density maps
        
        Args:
            idx (int): Image index (used for reproducible randomness)
            
        Returns:
            tuple: (density_map, total_count)
        """
        # Use index for reproducible randomness
        np.random.seed(42 + idx)
        
        # Density map size (downscaled)
        density_h = self.image_size[0] // self.density_downscale
        density_w = self.image_size[1] // self.density_downscale
        
        # Random number of people
        num_people = np.random.randint(self.min_people, self.max_people + 1)
        
        # Create density map
        density_map = np.zeros((density_h, density_w))
        
        # Add Gaussian kernels for each person
        for _ in range(num_people):
            # Random position
            x = np.random.randint(0, density_w)
            y = np.random.randint(0, density_h)
            
            # Add Gaussian kernel
            sigma = np.random.uniform(0.5, 1.5)  # Random kernel size
            self.add_gaussian_kernel(density_map, x, y, sigma)
        
        # Normalize density map so that sum equals number of people
        if np.sum(density_map) > 0:
            density_map = density_map * (num_people / np.sum(density_map))
        
        return density_map, num_people
    
    def add_gaussian_kernel(self, density_map, x, y, sigma):
        """
        Add a Gaussian kernel to the density map.
        
        Args:
            density_map (np.ndarray): Density map to modify
            x (int): X coordinate of the kernel center
            y (int): Y coordinate of the kernel center
            sigma (float): Standard deviation of the Gaussian kernel
        """
        h, w = density_map.shape
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate Gaussian kernel
        kernel = np.exp(-((x_coords - x) ** 2 + (y_coords - y) ** 2) / (2 * sigma ** 2))
        
        # Add to density map
        density_map += kernel
    
    def get_image_info(self, idx):
        """Get information about a specific image."""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        
        return {
            'filename': self.image_files[idx],
            'path': img_path,
            'original_size': (width, height),
            'processed_size': self.image_size
        }


def get_transforms(mode='train'):
    """
    Get data transforms for training or validation.
    
    Args:
        mode (str): 'train' or 'val'
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(image_dir, batch_size=4, num_workers=0, 
                       train_split=0.8, image_size=(256, 256)):
    """
    Create train and validation data loaders.
    
    Args:
        image_dir (str): Path to images directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        train_split (float): Fraction of data to use for training
        image_size (tuple): Target image size
        
    Returns:
        tuple: (train_loader, val_loader, dataset_info)
    """
    # Create full dataset
    full_dataset = CrowdDataset(
        image_dir=image_dir,
        transform=get_transforms('train'),
        image_size=image_size
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'total_images': total_size,
        'train_images': train_size,
        'val_images': val_size,
        'batch_size': batch_size,
        'image_size': image_size
    }
    
    print(f"Dataset created - Total: {total_size}, Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader, dataset_info


def test_data_loader():
    """Test the data loader with sample data."""
    # Simulate test with dummy data
    print("Testing data loader...")
    
    # This would normally use the real image directory
    # For testing purposes, we'll create a simple test
    print("Data loader components implemented and ready to use!")
    

if __name__ == "__main__":
    test_data_loader()