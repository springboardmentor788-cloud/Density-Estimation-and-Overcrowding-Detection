"""
Visualization script for crowd density estimation results.
Provides comprehensive visualization of images, density maps, and predictions.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# Import project modules
from models.csrnet import create_csrnet
from data_loader import create_data_loaders, CrowdDataset, get_transforms
from utils import (load_checkpoint, denormalize_image, apply_colormap_to_density, 
                  set_seed, resize_density_map)


class CrowdVisualizer:
    """
    Visualizer class for crowd density estimation results.
    Creates various types of visualizations for model outputs.
    """
    
    def __init__(self, model_path, config):
        """
        Initialize visualizer.
        
        Args:
            model_path (str): Path to trained model checkpoint
            config (dict): Visualization configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_seed(config['seed'])
        
        # Load model
        self.model = create_csrnet(pretrained=False)
        self.model.to(self.device)
        
        # Load trained weights
        if model_path and os.path.exists(model_path):
            load_checkpoint(model_path, self.model)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model checkpoint not found at {model_path}")
            print("Using randomly initialized model (for demonstration)")
        
        self.model.eval()
        
        # Create output directory
        self.output_dir = config.get('output_dir', './visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_single_prediction(self, image, density_pred, density_gt, count_pred, count_gt, 
                                   save_path=None, show_plot=True):
        """
        Visualize prediction for a single image.
        
        Args:
            image (torch.Tensor or np.ndarray): Input image
            density_pred (np.ndarray): Predicted density map
            density_gt (np.ndarray): Ground truth density map
            count_pred (float): Predicted count
            count_gt (float): Ground truth count
            save_path (str, optional): Path to save the visualization
            show_plot (bool): Whether to display the plot
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # CHW format
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # HWC format
                image_np = image.cpu().numpy()
            
            # Denormalize if normalized
            if image_np.min() < 0:  # Likely normalized
                # Approximate denormalization
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        else:
            image_np = image
        
        # Ensure image is in [0, 1] range
        if image_np.max() > 1:
            image_np = image_np / 255.0
        
        # Resize density maps to match image size for overlay
        target_size = (image_np.shape[1], image_np.shape[0])  # (width, height)
        density_pred_resized = resize_density_map(density_pred, target_size)
        density_gt_resized = resize_density_map(density_gt, target_size)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Crowd Density Estimation Results\nPredicted: {count_pred:.1f}, Ground Truth: {count_gt:.1f}, Error: {abs(count_pred - count_gt):.1f}', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Ground truth density map
        gt_colored = apply_colormap_to_density(density_gt, 'viridis')
        im1 = axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title(f'Ground Truth Density\nCount: {count_gt:.1f}', fontsize=14)
        axes[0, 1].axis('off')
        
        # Predicted density map
        pred_colored = apply_colormap_to_density(density_pred, 'viridis')
        im2 = axes[0, 2].imshow(pred_colored)
        axes[0, 2].set_title(f'Predicted Density\nCount: {count_pred:.1f}', fontsize=14)
        axes[0, 2].axis('off')
        
        # Overlay ground truth on image
        overlay_gt = self.create_overlay(image_np, density_gt_resized, alpha=0.6)
        axes[1, 0].imshow(overlay_gt)
        axes[1, 0].set_title('GT Density Overlay', fontsize=14)
        axes[1, 0].axis('off')
        
        # Overlay prediction on image
        overlay_pred = self.create_overlay(image_np, density_pred_resized, alpha=0.6)
        axes[1, 1].imshow(overlay_pred)
        axes[1, 1].set_title('Predicted Density Overlay', fontsize=14)
        axes[1, 1].axis('off')
        
        # Error map
        error_map = np.abs(density_pred - density_gt)
        error_colored = apply_colormap_to_density(error_map, 'Reds')
        axes[1, 2].imshow(error_colored)
        axes[1, 2].set_title(f'Absolute Error Map\nMAE: {np.sum(error_map):.2f}', fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_overlay(self, image, density_map, alpha=0.6, colormap='jet'):
        """
        Create overlay of density map on image.
        
        Args:
            image (np.ndarray): Original image
            density_map (np.ndarray): Density map
            alpha (float): Transparency of overlay
            colormap (str): Matplotlib colormap name
            
        Returns:
            np.ndarray: Overlaid image
        """
        # Normalize density map
        if density_map.max() > 0:
            density_norm = density_map / density_map.max()
        else:
            density_norm = density_map
        
        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        density_colored = cmap(density_norm)[:, :, :3]  # Remove alpha channel
        
        # Create overlay
        overlay = (1 - alpha) * image + alpha * density_colored
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def visualize_batch(self, data_loader, num_samples=8, save_dir=None):
        """
        Visualize predictions for a batch of images.
        
        Args:
            data_loader (DataLoader): Data loader
            num_samples (int): Number of samples to visualize
            save_dir (str, optional): Directory to save visualizations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"Visualizing {num_samples} samples...")
        
        with torch.no_grad():
            for batch_idx, (images, density_maps, counts) in enumerate(data_loader):
                if batch_idx * data_loader.batch_size >= num_samples:
                    break
                
                # Move to device
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Resize predictions to match ground truth
                if predictions.shape != density_maps.shape:
                    predictions = torch.nn.functional.interpolate(
                        predictions, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # Process each image in batch
                for i in range(images.size(0)):
                    if batch_idx * data_loader.batch_size + i >= num_samples:
                        break
                    
                    # Extract data
                    image = images[i]
                    pred_density = predictions[i].squeeze().cpu().numpy()
                    gt_density = density_maps[i].squeeze().cpu().numpy()
                    pred_count = np.sum(pred_density)
                    gt_count = counts[i].item()
                    
                    # Create visualization
                    save_path = None
                    if save_dir:
                        save_path = os.path.join(save_dir, f'prediction_{batch_idx}_{i}.png')
                    
                    self.visualize_single_prediction(
                        image, pred_density, gt_density, pred_count, gt_count,
                        save_path=save_path, show_plot=False
                    )
        
        print(f"Batch visualization completed!")
    
    def create_comparison_grid(self, data_loader, num_samples=16, save_path=None):
        """
        Create a grid comparison of multiple predictions.
        
        Args:
            data_loader (DataLoader): Data loader
            num_samples (int): Number of samples to include
            save_path (str, optional): Path to save the grid
        """
        print(f"Creating comparison grid for {num_samples} samples...")
        
        # Collect data
        images_list = []
        pred_densities = []
        gt_densities = []
        pred_counts = []
        gt_counts = []
        
        with torch.no_grad():
            for batch_idx, (images, density_maps, counts) in enumerate(data_loader):
                if len(images_list) >= num_samples:
                    break
                
                # Move to device
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Resize predictions
                if predictions.shape != density_maps.shape:
                    predictions = torch.nn.functional.interpolate(
                        predictions, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # Collect data
                for i in range(images.size(0)):
                    if len(images_list) >= num_samples:
                        break
                    
                    images_list.append(images[i].cpu())
                    pred_densities.append(predictions[i].squeeze().cpu().numpy())
                    gt_densities.append(density_maps[i].squeeze().cpu().numpy())
                    pred_counts.append(np.sum(predictions[i].squeeze().cpu().numpy()))
                    gt_counts.append(counts[i].item())
        
        # Create grid
        rows = int(np.ceil(np.sqrt(len(images_list))))
        cols = int(np.ceil(len(images_list) / rows))
        
        fig, axes = plt.subplots(4, cols, figsize=(4*cols, 16))
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(len(images_list)):
            col = i % cols
            if i >= cols:
                continue  # Only show first row for grid
            
            # Convert image
            image = images_list[i]
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.numpy()
            
            if image_np.min() < 0:  # Denormalize
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            # Original image
            axes[0, col].imshow(image_np)
            axes[0, col].set_title(f'Sample {i+1}', fontsize=10)
            axes[0, col].axis('off')
            
            # Ground truth
            gt_colored = apply_colormap_to_density(gt_densities[i], 'viridis')
            axes[1, col].imshow(gt_colored)
            axes[1, col].set_title(f'GT: {gt_counts[i]:.1f}', fontsize=10)
            axes[1, col].axis('off')
            
            # Prediction
            pred_colored = apply_colormap_to_density(pred_densities[i], 'viridis')
            axes[2, col].imshow(pred_colored)
            axes[2, col].set_title(f'Pred: {pred_counts[i]:.1f}', fontsize=10)
            axes[2, col].axis('off')
            
            # Error
            error_map = np.abs(pred_densities[i] - gt_densities[i])
            error_colored = apply_colormap_to_density(error_map, 'Reds')
            axes[3, col].imshow(error_colored)
            axes[3, col].set_title(f'MAE: {np.sum(error_map):.1f}', fontsize=10)
            axes[3, col].axis('off')
        
        # Hide unused axes
        for col in range(len(images_list), cols):
            for row in range(4):
                axes[row, col].axis('off')
        
        # Add row labels
        fig.text(0.02, 0.875, 'Original', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.625, 'Ground Truth', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.375, 'Prediction', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.125, 'Error Map', rotation=90, fontsize=12, fontweight='bold', va='center')
        
        plt.suptitle('Crowd Density Estimation - Grid Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        
        plt.show()
    
    def plot_count_correlation(self, data_loader, save_path=None):
        """
        Plot correlation between predicted and ground truth counts.
        
        Args:
            data_loader (DataLoader): Data loader
            save_path (str, optional): Path to save the plot
        """
        print("Creating count correlation plot...")
        
        pred_counts = []
        gt_counts = []
        
        with torch.no_grad():
            for images, density_maps, counts in tqdm(data_loader, desc="Collecting predictions"):
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                predictions = self.model(images)
                
                for i in range(images.size(0)):
                    pred_count = torch.sum(predictions[i]).item()
                    gt_count = counts[i].item()
                    
                    pred_counts.append(pred_count)
                    gt_counts.append(gt_count)
        
        # Create correlation plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(gt_counts, pred_counts, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(min(gt_counts), min(pred_counts))
        max_val = max(max(gt_counts), max(pred_counts))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate correlation
        correlation = np.corrcoef(pred_counts, gt_counts)[0, 1]
        
        # Add statistics
        mae = np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts)))
        mse = np.mean((np.array(pred_counts) - np.array(gt_counts)) ** 2)
        
        plt.xlabel('Ground Truth Count', fontsize=12)
        plt.ylabel('Predicted Count', fontsize=12)
        plt.title(f'Count Prediction Correlation\nCorr: {correlation:.3f}, MAE: {mae:.2f}, MSE: {mse:.2f}', 
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Correlation plot saved to {save_path}")
        
        plt.show()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize CSRNet crowd density estimation results')
    parser.add_argument('--model_path', type=str, default='./outputs/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, default='../images',
                       help='Path to images directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'image_size': (256, 256),
        'output_dir': args.output_dir,
        'seed': 42,
        'num_workers': 0
    }
    
    print("Visualization Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples}")
    
    # Create data loader
    print("\nCreating data loaders...")
    _, val_loader, dataset_info = create_data_loaders(
        image_dir=args.image_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    # Create visualizer
    visualizer = CrowdVisualizer(args.model_path, config)
    
    # Create different types of visualizations
    print("\n1. Creating individual sample visualizations...")
    visualizer.visualize_batch(
        val_loader, 
        num_samples=args.num_samples, 
        save_dir=os.path.join(args.output_dir, 'individual_samples')
    )
    
    print("\n2. Creating comparison grid...")
    visualizer.create_comparison_grid(
        val_loader,
        num_samples=min(16, args.num_samples),
        save_path=os.path.join(args.output_dir, 'comparison_grid.png')
    )
    
    print("\n3. Creating count correlation plot...")
    visualizer.plot_count_correlation(
        val_loader,
        save_path=os.path.join(args.output_dir, 'count_correlation.png')
    )
    
    print(f"\nVisualization completed! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()