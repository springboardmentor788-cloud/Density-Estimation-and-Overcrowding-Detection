import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split

from dataset import CrowdDataset
from models.csrnet import CSRNet
import config


def evaluate():

    print("Starting evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CrowdDataset(config.IMAGE_DIR, config.GT_DIR)

    # Same split (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Load model
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print("✅ Model loaded successfully!")

    gt_counts = []
    pred_counts = []

    with torch.no_grad():

        for idx, (images, density_maps) in enumerate(val_loader):

            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)

            # Counts
            pred_count = outputs.sum().item()
            gt_count = density_maps.sum().item()

            pred_counts.append(pred_count)
            gt_counts.append(gt_count)

            print(f"\nSample {idx+1}")
            print(f"Predicted: {pred_count:.2f} | GT: {gt_count:.2f}")

    # Convert to numpy
    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)

    # 🔥 Metrics
    r2 = r2_score(gt_counts, pred_counts)
    mae = mean_absolute_error(gt_counts, pred_counts)
    mse = mean_squared_error(gt_counts, pred_counts)

    print("\n📊 FINAL METRICS")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

    # 🔥 Scatter Plot (R² Graph)
    plt.figure()
    plt.scatter(gt_counts, pred_counts, alpha=0.6)

    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title(f"R² = {r2:.4f}")

    # Ideal line
    min_val = min(gt_counts.min(), pred_counts.min())
    max_val = max(gt_counts.max(), pred_counts.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.grid()
    plt.savefig("r2_scatter.png")

    print("📊 R2 scatter plot saved as r2_scatter.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
