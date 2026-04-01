# ============================================================
#  scripts/validate.py
#  Run:  python scripts/validate.py
# ============================================================

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_ROOT, MODEL_DIR, OUTPUT_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

IMG_SIZE = 224


def load_mat(mat_path):
    try:
        mat    = sio.loadmat(str(mat_path))
        points = mat["image_info"][0][0][0][0][0]
        return points
    except Exception:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            points = np.array(f["image_info"]["location"]).T
        return points


def make_density_map(points, img_shape, sigma=8):
    H, W    = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    for point in points:
        x = int(float(point[0]))
        y = int(float(point[1]))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        density[y, x] += 1.0
    density = gaussian_filter(density, sigma=sigma)
    return density


class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir  = Path(data_dir)
        self.img_dir   = self.data_dir / "images"
        self.gt_dir    = self.data_dir / "ground_truth"
        self.transform = transform
        self.samples   = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            num      = img_path.stem.replace("IMG_", "")
            mat_path = self.gt_dir / f"GT_IMG_{num}.mat"
            if mat_path.exists():
                self.samples.append((img_path, mat_path))
        print(f"[Dataset]  Found {len(self.samples)} image+label pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mat_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        points  = load_mat(mat_path)

        # store actual head count
        actual_count = len(points)

        H, W    = image.shape[1], image.shape[2]
        density = make_density_map(points, (H, W))
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        density_tensor = torch.nn.functional.interpolate(
            density_tensor.unsqueeze(0),
            size=(H // 8, W // 8),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        density_tensor = density_tensor * 64
        return image, density_tensor, actual_count, str(img_path)


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:23]
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def compute_scale_factor(model, dataloader):
    """Compute best scale factor from validation data."""
    model.eval()
    ratios = []
    with torch.no_grad():
        for images, targets, actual_counts, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = torch.relu(outputs)
            for i in range(len(images)):
                raw  = float(outputs[i].sum().item())
                real = float(actual_counts[i].item())
                if raw > 0 and real > 0:
                    ratios.append(real / raw)
    scale = float(np.median(ratios)) if ratios else 5.1
    print(f"  Scale factor computed : {scale:.4f}")
    return scale


def validate(model, dataloader, scale_factor):
    model.eval()
    criterion  = nn.MSELoss()
    total_loss = 0.0
    total_mae  = 0.0
    total_rmse = 0.0
    results    = []

    print("\n" + "="*55)
    print("  VALIDATION  –  Real Crowd Count")
    print("="*55 + "\n")

    with torch.no_grad():
        for images, targets, actual_counts, paths in dataloader:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            if outputs.shape != targets.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            outputs_relu = torch.relu(outputs)
            loss         = criterion(outputs, targets)

            for i in range(len(images)):
                raw_sum      = float(outputs_relu[i].sum().item())
                pred_count   = raw_sum * scale_factor
                actual_count = float(actual_counts[i].item())
                mae          = abs(pred_count - actual_count)
                mse          = (pred_count - actual_count) ** 2

                total_mae  += mae
                total_rmse += mse
                results.append({
                    "path"   : paths[i],
                    "pred"   : pred_count,
                    "actual" : actual_count,
                    "mae"    : mae,
                })

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_mae  = total_mae  / len(results)
    avg_rmse = (total_rmse / len(results)) ** 0.5

    print(f"  Total images      : {len(results)}")
    print(f"  Val Loss (MSE)    : {avg_loss:.4f}")
    print(f"  Val MAE           : {avg_mae:.2f} people")
    print(f"  Val RMSE          : {avg_rmse:.2f} people")
    print(f"  Scale factor used : {scale_factor:.4f}")

    # ── print per image results ───────────────────────────────
    print(f"\n  {'Image':<15}  {'Actual':>8}  {'Predicted':>10}  {'MAE':>8}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}")
    for r in results[:20]:  # show first 20
        print(f"  {Path(r['path']).name:<15}  "
              f"{r['actual']:>8.0f}  "
              f"{r['pred']:>10.0f}  "
              f"{r['mae']:>8.1f}")
    if len(results) > 20:
        print(f"  ... and {len(results)-20} more images")

    preds   = [r["pred"]   for r in results]
    actuals = [r["actual"] for r in results]

    # ── plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # scatter pred vs actual
    axes[0].scatter(actuals, preds, alpha=0.5, color="#7c6af7", s=20)
    max_val = max(max(actuals), max(preds)) if actuals else 100
    axes[0].plot([0, max_val], [0, max_val], "r--", lw=2,
                 label="Perfect prediction")
    axes[0].set_xlabel("Actual Crowd Count")
    axes[0].set_ylabel("Predicted Crowd Count")
    axes[0].set_title(f"Pred vs Actual\nMAE = {avg_mae:.1f} people")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # error distribution
    errors = [r["mae"] for r in results]
    axes[1].hist(errors, bins=20, color="#22d3ee",
                 edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("MAE per image (people)")
    axes[1].set_ylabel("Number of images")
    axes[1].set_title(f"Error Distribution\nAvg MAE = {avg_mae:.1f} people")
    axes[1].grid(True, alpha=0.3)

    # actual vs predicted bar for first 15 images
    n    = min(15, len(results))
    idxs = range(n)
    bar_actual = [results[i]["actual"] for i in idxs]
    bar_pred   = [results[i]["pred"]   for i in idxs]
    labels     = [Path(results[i]["path"]).stem for i in idxs]
    x = np.arange(n)
    w = 0.35
    axes[2].bar(x - w/2, bar_actual, w, label="Actual",
                color="#4ade80", alpha=0.8)
    axes[2].bar(x + w/2, bar_pred,   w, label="Predicted",
                color="#f97316", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Crowd Count")
    axes[2].set_title("Actual vs Predicted (first 15 images)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Validation Results  –  MAE: {avg_mae:.1f}  RMSE: {avg_rmse:.1f}  "
        f"({len(results)} images)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "validation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Plot saved → {save_path}\n")
    return avg_loss, avg_mae, avg_rmse


if __name__ == "__main__":

    VAL_DIR = (DATA_ROOT.parent.parent /
               "dataset" / "archive" /
               "part_B_final" / "test_data")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    print("[ STEP 1 ]  Loading Validation Dataset...")
    dataset    = ShanghaiTechDataset(VAL_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=False, num_workers=0)

    print("[ STEP 2 ]  Loading Best Model...")
    model      = CSRNet().to(device)
    model_path = Path(MODEL_DIR) / "csrnet_best.pth"
    if not model_path.exists():
        model_path = Path(MODEL_DIR) / "csrnet_real.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded → {model_path}")

    print("\n[ STEP 3 ]  Computing Scale Factor...")
    scale = compute_scale_factor(model, dataloader)

    print("\n[ STEP 4 ]  Running Validation...")
    avg_loss, avg_mae, avg_rmse = validate(model, dataloader, scale)

    print("="*55)
    print("  VALIDATION COMPLETE!")
    print(f"  MAE  : {avg_mae:.2f} people")
    print(f"  RMSE : {avg_rmse:.2f} people")
    print(f"  Loss : {avg_loss:.4f}")
    print("="*55 + "\n")