# ============================================================
#  scripts/test.py
#  Run:  python scripts/test.py
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
        H, W    = image.shape[1], image.shape[2]
        density = make_density_map(points, (H, W))
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        density_tensor = torch.nn.functional.interpolate(
            density_tensor.unsqueeze(0),
            size=(H // 8, W // 8),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        density_tensor = density_tensor * 64
        return image, density_tensor, str(img_path)


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:23]
        )
        # ── 4 layers  – must match train.py ──────────────────
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


def test(model, dataloader):
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    results   = []

    print("\n" + "="*50)
    print("  FINAL TESTING")
    print("="*50 + "\n")

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            if outputs.shape != targets.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            pred_count   = outputs.sum().item() / 64
            target_count = targets.sum().item() / 64
            mae          = abs(pred_count - target_count)
            mse          = (pred_count - target_count) ** 2
            total_mae   += mae
            total_mse   += mse
            results.append({
                "path"   : paths[0],
                "pred"   : pred_count,
                "actual" : target_count,
                "mae"    : mae,
            })
            print(f"  {Path(paths[0]).name:<15}  "
                  f"Actual: {target_count:>6.0f}  "
                  f"Predicted: {pred_count:>6.0f}  "
                  f"MAE: {mae:.1f}")

    avg_mae  = total_mae / len(dataloader)
    avg_rmse = (total_mse / len(dataloader)) ** 0.5

    print("\n" + "-"*50)
    print(f"  Final MAE  : {avg_mae:.2f}")
    print(f"  Final RMSE : {avg_rmse:.2f}")
    print(f"  Total images : {len(results)}")

    preds   = [r["pred"]   for r in results]
    actuals = [r["actual"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(actuals, preds, alpha=0.5, color="#22d3ee", s=20)
    max_val = max(max(actuals), max(preds)) if actuals else 100
    axes[0].plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Actual Count")
    axes[0].set_ylabel("Predicted Count")
    axes[0].set_title("Test  –  Pred vs Actual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    errors = [r["mae"] for r in results]
    axes[1].hist(errors, bins=20, color="#f97316",
                 edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("MAE per image")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error Distribution (Avg MAE={avg_mae:.1f})")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Final Test Results  –  MAE:{avg_mae:.1f}  RMSE:{avg_rmse:.1f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "test_results.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved → {save_path}\n")
    return avg_mae, avg_rmse


if __name__ == "__main__":

    TEST_DIR = (DATA_ROOT.parent.parent /
                "dataset" / "archive" /
                "part_A_final" / "test_data")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    print("[ STEP 1 ]  Loading Test Dataset...")
    dataset    = ShanghaiTechDataset(TEST_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    print("[ STEP 2 ]  Loading Best Model...")
    model      = CSRNet().to(device)
    model_path = Path(MODEL_DIR) / "csrnet_best.pth"
    if not model_path.exists():
        model_path = Path(MODEL_DIR) / "csrnet_real.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded → {model_path}")

    print("\n[ STEP 3 ]  Running Final Test...")
    avg_mae, avg_rmse = test(model, dataloader)

    print("="*50)
    print("  TESTING COMPLETE!")
    print(f"  Final MAE  : {avg_mae:.2f}")
    print(f"  Final RMSE : {avg_rmse:.2f}")
    print("="*50 + "\n")