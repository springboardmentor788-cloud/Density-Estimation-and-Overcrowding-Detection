"""
Utility functions for crowd density estimation project.
Contains helper functions for metrics calculation, visualization, and model utilities.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import os


def calculate_mae(pred_density, gt_density):
    """
    Calculate Mean Absolute Error between predicted and ground truth density maps.
    
    Args:
        pred_density (torch.Tensor or np.ndarray): Predicted density map
        gt_density (torch.Tensor or np.ndarray): Ground truth density map
        
    Returns:
        float: MAE value
    """
    if isinstance(pred_density, torch.Tensor):
        pred_count = torch.sum(pred_density).item()
    else:
        pred_count = np.sum(pred_density)
        
    if isinstance(gt_density, torch.Tensor):
        gt_count = torch.sum(gt_density).item()
    else:
        gt_count = np.sum(gt_density)
    
    mae = abs(pred_count - gt_count)
    return mae


def calculate_mse(pred_density, gt_density):
    """
    Calculate Mean Squared Error between predicted and ground truth density maps.
    
    Args:
        pred_density (torch.Tensor or np.ndarray): Predicted density map
        gt_density (torch.Tensor or np.ndarray): Ground truth density map
        
    Returns:
        float: MSE value
    """
    if isinstance(pred_density, torch.Tensor):
        pred_count = torch.sum(pred_density).item()
    else:
        pred_count = np.sum(pred_density)
        
    if isinstance(gt_density, torch.Tensor):
        gt_count = torch.sum(gt_density).item()
    else:
        gt_count = np.sum(gt_density)
    
    mse = (pred_count - gt_count) ** 2
    return mse


def calculate_pixel_wise_mse(pred_density, gt_density):
    """
    Calculate pixel-wise Mean Squared Error between density maps.
    
    Args:
        pred_density (torch.Tensor): Predicted density map
        gt_density (torch.Tensor): Ground truth density map
        
    Returns:
        float: Pixel-wise MSE value
    """
    if isinstance(pred_density, torch.Tensor) and isinstance(gt_density, torch.Tensor):
        mse = torch.mean((pred_density - gt_density) ** 2).item()
    else:
        # Convert to numpy if needed
        if isinstance(pred_density, torch.Tensor):
            pred_density = pred_density.detach().cpu().numpy()
        if isinstance(gt_density, torch.Tensor):
            gt_density = gt_density.detach().cpu().numpy()
        mse = np.mean((pred_density - gt_density) ** 2)
    
    return mse


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics.
        
        Args:
            val (float): Current value
            n (int): Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to the checkpoint file
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        
    Returns:
        int: Epoch number from checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file {filepath} not found!")
        return 0
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): Model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (list): Normalization mean values
        std (list): Normalization std values
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def apply_colormap_to_density(density_map, colormap='jet'):
    """
    Apply colormap to density map for visualization.
    
    Args:
        density_map (np.ndarray): Density map array
        colormap (str): Matplotlib colormap name
        
    Returns:
        np.ndarray: Colored density map
    """
    # Normalize density map to 0-255
    normalized = ((density_map - density_map.min()) / 
                  (density_map.max() - density_map.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    colored = cmap(normalized / 255.0)
    
    # Convert to RGB (remove alpha channel)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_rgb


def resize_density_map(density_map, target_size):
    """
    Resize density map while preserving total count.
    
    Args:
        density_map (np.ndarray): Original density map
        target_size (tuple): Target size (height, width)
        
    Returns:
        np.ndarray: Resized density map
    """
    original_count = np.sum(density_map)
    
    # Resize using interpolation
    resized = cv2.resize(density_map, target_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    # Preserve total count
    if np.sum(resized) > 0:
        resized = resized * (original_count / np.sum(resized))
    
    return resized


def create_output_directories(base_dir):
    """
    Create necessary output directories for saving results.
    
    Args:
        base_dir (str): Base directory path
        
    Returns:
        dict: Dictionary of created directory paths
    """
    directories = {
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    return directories


def print_model_summary(model, input_size=(1, 3, 256, 256)):
    """
    Print model architecture summary.
    
    Args:
        model (torch.nn.Module): Model to summarize
        input_size (tuple): Input tensor size
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        # Get device of model parameters
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        output = model(dummy_input)
        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print(f"Device: {device}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i * 2)
    print(f"Average: {meter.avg}")
    
    # Test metrics
    pred = np.random.rand(32, 32)
    gt = np.random.rand(32, 32)
    mae = calculate_mae(pred, gt)
    mse = calculate_mse(pred, gt)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    print("All utility functions working correctly!")