import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from dataset import CrowdDataset
from models.csrnet import CSRNet
import config


def evaluate_r2(num_samples=30):

    print("Running R² Evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load Dataset
    # ----------------------------
    dataset = CrowdDataset(config.IMAGE_DIR, config.GT_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ----------------------------
    # Load Model
    # ----------------------------
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print("Model loaded successfully!")

    gt_counts = []
    pred_counts = []

    # ----------------------------
    # Inference
    # ----------------------------
    with torch.no_grad():

        for idx, (images, density_maps) in enumerate(val_loader):

            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)

            pred_count = outputs.sum().item()
            gt_count = density_maps.sum().item()

            gt_counts.append(gt_count)
            pred_counts.append(pred_count)

            print(f"Sample {idx+1}: GT={gt_count:.2f}, Pred={pred_count:.2f}")

            if idx + 1 >= num_samples:
                break

    # ----------------------------
    # Metrics
    # ----------------------------
    r2 = r2_score(gt_counts, pred_counts)
    mae = mean_absolute_error(gt_counts, pred_counts)
    mse = mean_squared_error(gt_counts, pred_counts)

    print("\n📊 Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # ----------------------------
    # Plot R² Graph
    # ----------------------------
    plot_r2_graph(gt_counts, pred_counts)

    # ----------------------------
    # Save Results
    # ----------------------------
    np.save("gt_counts.npy", np.array(gt_counts))
    np.save("pred_counts.npy", np.array(pred_counts))

    print("✅ Results saved!")


def plot_r2_graph(gt, pred):

    plt.figure()

    plt.scatter(gt, pred)

    # Ideal line (y = x)
    min_val = min(gt)
    max_val = max(gt)

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("R² Graph (GT vs Pred)")

    plt.savefig("r2_plot.png")

    print("📊 R² graph saved as r2_plot.png")

    plt.show()


if __name__ == "__main__":
    evaluate_r2(num_samples=30)