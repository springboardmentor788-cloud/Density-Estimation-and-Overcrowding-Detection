"""
Evaluation script for trained CSRNet crowd density estimation model.
Evaluates model performance and generates detailed metrics reports.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import project modules
from models.csrnet import create_csrnet
from data_loader import create_data_loaders, CrowdDataset, get_transforms
from utils import (AverageMeter, calculate_mae, calculate_mse, calculate_pixel_wise_mse,
                  load_checkpoint, set_seed, apply_colormap_to_density)


class CrowdEvaluator:
    """
    Evaluator class for crowd density estimation model.
    Handles model evaluation and metrics computation.
    """
    
    def __init__(self, model_path, config):
        """
        Initialize evaluator.
        
        Args:
            model_path (str): Path to trained model checkpoint
            config (dict): Evaluation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
        
        # Results storage
        self.results = {
            'individual_results': [],
            'summary_metrics': {},
            'config': config,
            'evaluation_time': datetime.now().isoformat()
        }
    
    def evaluate_dataset(self, data_loader, dataset_name="Test"):
        """
        Evaluate model on entire dataset.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            dataset_name (str): Name of the dataset being evaluated
            
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating on {dataset_name} dataset...")
        
        # Initialize metrics
        maes = AverageMeter()
        mses = AverageMeter()
        pixel_mses = AverageMeter()
        losses = AverageMeter()
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Store individual results
        individual_results = []
        predicted_counts = []
        ground_truth_counts = []
        
        with torch.no_grad():
            for batch_idx, (images, density_maps, counts) in enumerate(tqdm(data_loader, desc="Evaluating")):
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
                loss = criterion(outputs, density_maps)
                
                # Process each image in the batch
                for i in range(outputs.size(0)):
                    # Extract individual predictions and ground truth
                    pred_density = outputs[i].squeeze().cpu().numpy()
                    gt_density = density_maps[i].squeeze().cpu().numpy()
                    gt_count = counts[i].item()
                    
                    # Calculate metrics
                    pred_count = np.sum(pred_density)
                    mae = calculate_mae(pred_density, gt_density)
                    mse = calculate_mse(pred_density, gt_density)
                    pixel_mse = calculate_pixel_wise_mse(outputs[i], density_maps[i])
                    
                    # Store results
                    result = {
                        'image_idx': batch_idx * data_loader.batch_size + i,
                        'predicted_count': float(pred_count),
                        'ground_truth_count': float(gt_count),
                        'mae': float(mae),
                        'mse': float(mse),
                        'pixel_mse': float(pixel_mse),
                        'relative_error': float(abs(pred_count - gt_count) / max(gt_count, 1)) * 100
                    }
                    
                    individual_results.append(result)
                    predicted_counts.append(pred_count)
                    ground_truth_counts.append(gt_count)
                    
                    # Update average meters
                    maes.update(mae)
                    mses.update(mse)
                    pixel_mses.update(pixel_mse)
                
                losses.update(loss.item(), images.size(0))
        
        # Calculate summary statistics
        predicted_counts = np.array(predicted_counts)
        ground_truth_counts = np.array(ground_truth_counts)
        
        # Correlation coefficient
        correlation = np.corrcoef(predicted_counts, ground_truth_counts)[0, 1]
        
        # Additional metrics
        relative_errors = np.abs(predicted_counts - ground_truth_counts) / np.maximum(ground_truth_counts, 1) * 100
        mean_relative_error = np.mean(relative_errors)
        
        summary_results = {
            'dataset_name': dataset_name,
            'num_samples': len(individual_results),
            'average_mae': maes.avg,
            'average_mse': mses.avg,
            'average_pixel_mse': pixel_mses.avg,
            'average_loss': losses.avg,
            'correlation_coefficient': correlation if not np.isnan(correlation) else 0.0,
            'mean_relative_error': mean_relative_error,
            'mean_predicted_count': np.mean(predicted_counts),
            'mean_ground_truth_count': np.mean(ground_truth_counts),
            'std_predicted_count': np.std(predicted_counts),
            'std_ground_truth_count': np.std(ground_truth_counts),
            'min_mae': min([r['mae'] for r in individual_results]),
            'max_mae': max([r['mae'] for r in individual_results]),
            'median_mae': np.median([r['mae'] for r in individual_results])
        }
        
        return summary_results, individual_results
    
    def generate_detailed_report(self, summary_results, individual_results):
        """
        Generate and print detailed evaluation report.
        
        Args:
            summary_results (dict): Summary metrics
            individual_results (list): Individual sample results
        """
        print("\n" + "="*60)
        print("DETAILED EVALUATION REPORT")
        print("="*60)
        
        print(f"Dataset: {summary_results['dataset_name']}")
        print(f"Number of samples: {summary_results['num_samples']}")
        print(f"Evaluation time: {self.results['evaluation_time']}")
        
        print(f"\nCOUNT ACCURACY METRICS:")
        print(f"  Mean Absolute Error (MAE): {summary_results['average_mae']:.2f}")
        print(f"  Mean Squared Error (MSE): {summary_results['average_mse']:.2f}")
        print(f"  Mean Relative Error: {summary_results['mean_relative_error']:.2f}%")
        print(f"  Correlation Coefficient: {summary_results['correlation_coefficient']:.4f}")
        
        print(f"\nDENSITY MAP METRICS:")
        print(f"  Pixel-wise MSE: {summary_results['average_pixel_mse']:.6f}")
        print(f"  Average Loss: {summary_results['average_loss']:.6f}")
        
        print(f"\nCOUNT STATISTICS:")
        print(f"  Mean Predicted Count: {summary_results['mean_predicted_count']:.2f}")
        print(f"  Mean Ground Truth Count: {summary_results['mean_ground_truth_count']:.2f}")
        print(f"  Std Predicted Count: {summary_results['std_predicted_count']:.2f}")
        print(f"  Std Ground Truth Count: {summary_results['std_ground_truth_count']:.2f}")
        
        print(f"\nMAE DISTRIBUTION:")
        print(f"  Minimum MAE: {summary_results['min_mae']:.2f}")
        print(f"  Maximum MAE: {summary_results['max_mae']:.2f}")
        print(f"  Median MAE: {summary_results['median_mae']:.2f}")
        
        # Error distribution analysis
        errors = [r['mae'] for r in individual_results]
        print(f"\nERROR DISTRIBUTION:")
        print(f"  Samples with MAE < 5: {len([e for e in errors if e < 5])} ({len([e for e in errors if e < 5])/len(errors)*100:.1f}%)")
        print(f"  Samples with MAE < 10: {len([e for e in errors if e < 10])} ({len([e for e in errors if e < 10])/len(errors)*100:.1f}%)")
        print(f"  Samples with MAE > 20: {len([e for e in errors if e > 20])} ({len([e for e in errors if e > 20])/len(errors)*100:.1f}%)")
        
        print("="*60)
    
    def save_results(self, output_dir):
        """
        Save evaluation results to files.
        
        Args:
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results
        json_safe_results = convert_numpy_types(self.results)
        
        # Save detailed results as JSON
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        # Save summary as text
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CROWD DENSITY ESTIMATION - EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in self.results['summary_metrics'].items():
                if isinstance(value, (float, np.floating)):
                    f.write(f"{key}: {float(value):.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Summary saved to {summary_path}")
    
    def run_evaluation(self, data_loader, dataset_name="Test", save_results=True, output_dir="./eval_outputs"):
        """
        Run complete evaluation pipeline.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            dataset_name (str): Name of the dataset
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            
        Returns:
            dict: Complete evaluation results
        """
        # Evaluate model
        summary_results, individual_results = self.evaluate_dataset(data_loader, dataset_name)
        
        # Store results
        self.results['summary_metrics'] = summary_results
        self.results['individual_results'] = individual_results
        
        # Generate report
        self.generate_detailed_report(summary_results, individual_results)
        
        # Save results if requested
        if save_results:
            self.save_results(output_dir)
        
        return self.results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate CSRNet crowd density estimation model')
    parser.add_argument('--model_path', type=str, default='./outputs/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, default='../images',
                       help='Path to test images directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'image_size': (256, 256),
        'seed': 42,
        'num_workers': 0
    }
    
    print("Evaluation Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Output directory: {args.output_dir}")
    
    # Create data loader for evaluation
    print("\nCreating data loaders...")
    _, val_loader, dataset_info = create_data_loaders(
        image_dir=args.image_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size'],
        train_split=0.8  # We'll use validation set for evaluation
    )
    
    print(f"Evaluation dataset size: {dataset_info['val_images']} images")
    
    # Create evaluator
    evaluator = CrowdEvaluator(args.model_path, config)
    
    # Run evaluation
    results = evaluator.run_evaluation(
        val_loader, 
        dataset_name="Validation",
        save_results=True,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation completed!")
    print(f"Key Metrics:")
    print(f"  - MAE: {results['summary_metrics']['average_mae']:.2f}")
    print(f"  - MSE: {results['summary_metrics']['average_mse']:.2f}")
    print(f"  - Correlation: {results['summary_metrics']['correlation_coefficient']:.4f}")


if __name__ == "__main__":
    main()