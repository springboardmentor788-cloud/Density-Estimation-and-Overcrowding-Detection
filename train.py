"""
Training script for CSRNet crowd density estimation model.
Implements complete training pipeline with validation and model checkpointing.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import time

# Import project modules
from models.csrnet import create_csrnet
from data_loader import create_data_loaders
from utils import (AverageMeter, calculate_mae, calculate_mse, calculate_pixel_wise_mse,
                  save_checkpoint, load_checkpoint, set_seed, print_model_summary,
                  create_output_directories)


class CrowdTrainer:
    """
    Trainer class for crowd density estimation model.
    Handles training loop, validation, and metrics tracking.
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed(config['seed'])
        
        # Create output directories
        self.output_dirs = create_output_directories('./outputs')
        
        # Initialize model
        self.model = create_csrnet(pretrained=config['pretrained'])
        self.model.to(self.device)
        
        # Print model summary
        print_model_summary(self.model)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['scheduler_step'],
            gamma=config['scheduler_gamma']
        )
        
        # Initialize metrics tracking
        self.best_mae = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_mses = []
        
        # Load checkpoint if resuming
        self.start_epoch = 0
        if config['resume_checkpoint']:
            self.start_epoch = load_checkpoint(
                config['resume_checkpoint'], 
                self.model, 
                self.optimizer
            )
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        
        # Initialize metrics
        losses = AverageMeter()
        maes = AverageMeter()
        batch_time = AverageMeter()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        start_time = time.time()
        
        for batch_idx, (images, density_maps, counts) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Resize outputs to match ground truth size if needed
            if outputs.shape != density_maps.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                )
            
            # Calculate loss
            loss = self.criterion(outputs, density_maps)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            batch_mae = sum([calculate_mae(outputs[i], density_maps[i]) 
                           for i in range(outputs.size(0))]) / outputs.size(0)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            maes.update(batch_mae, images.size(0))
            
            # Update progress bar
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'MAE': f'{maes.avg:.2f}',
                'Time': f'{batch_time.avg:.2f}s'
            })
        
        return losses.avg
    
    def validate(self, val_loader, epoch):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            epoch (int): Current epoch number
            
        Returns:
            tuple: (avg_loss, avg_mae, avg_mse)
        """
        self.model.eval()
        
        # Initialize metrics
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        pixel_mses = AverageMeter()
        
        with torch.no_grad():
            for images, density_maps, counts in tqdm(val_loader, desc='Validating'):
                # Move data to device
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Resize outputs to match ground truth size if needed
                if outputs.shape != density_maps.shape:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # Calculate loss
                loss = self.criterion(outputs, density_maps)
                
                # Calculate metrics for each image in batch
                batch_mae = 0
                batch_mse = 0
                batch_pixel_mse = 0
                
                for i in range(outputs.size(0)):
                    mae = calculate_mae(outputs[i], density_maps[i])
                    mse = calculate_mse(outputs[i], density_maps[i])
                    pixel_mse = calculate_pixel_wise_mse(outputs[i], density_maps[i])
                    
                    batch_mae += mae
                    batch_mse += mse
                    batch_pixel_mse += pixel_mse
                
                # Average over batch
                batch_mae /= outputs.size(0)
                batch_mse /= outputs.size(0)
                batch_pixel_mse /= outputs.size(0)
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
                maes.update(batch_mae, images.size(0))
                mses.update(batch_mse, images.size(0))
                pixel_mses.update(batch_pixel_mse, images.size(0))
        
        print(f"\nValidation Results - Epoch {epoch+1}:")
        print(f"Loss: {losses.avg:.4f}")
        print(f"MAE: {maes.avg:.2f}")
        print(f"MSE: {mses.avg:.2f}")
        print(f"Pixel MSE: {pixel_mses.avg:.6f}")
        
        return losses.avg, maes.avg, mses.avg
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        """
        print(f"\nStarting training from epoch {self.start_epoch + 1}")
        print(f"Total epochs: {self.config['num_epochs']}")
        print("=" * 50)
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_mae, val_mse = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            self.val_mses.append(val_mse)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, MSE: {val_mse:.2f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if best model
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                checkpoint_path = os.path.join(self.output_dirs['checkpoints'], 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
                print(f"New best model saved! MAE: {val_mae:.2f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                checkpoint_path = os.path.join(self.output_dirs['checkpoints'], f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
            
            print("-" * 50)
        
        print(f"\nTraining completed!")
        print(f"Best validation MAE: {self.best_mae:.2f}")


def get_default_config():
    """Get default training configuration."""
    return {
        'image_dir': '../images',  # Path to image directory
        'batch_size': 4,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler_step': 10,
        'scheduler_gamma': 0.5,
        'image_size': (256, 256),
        'save_freq': 5,
        'pretrained': True,
        'resume_checkpoint': None,
        'seed': 42,
        'num_workers': 0
    }


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CSRNet for crowd density estimation')
    parser.add_argument('--image_dir', type=str, default='../images', 
                       help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    config['image_dir'] = args.image_dir
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['resume_checkpoint'] = args.resume
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, dataset_info = create_data_loaders(
        image_dir=config['image_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    print(f"Dataset Info: {dataset_info}")
    
    # Create trainer
    trainer = CrowdTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()