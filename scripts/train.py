# ============================================================
#  scripts/train.py
#  Run:  python scripts/train.py --epochs 50 --batch 8
# ============================================================

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import csv

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_ROOT, MODEL_DIR, OUTPUT_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

IMG_SIZE = 224   # faster than 224 on CPU


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


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
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
        self._init_weights()

    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def save_losses_csv(loss_log, output_dir):
    out      = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_mae", "val_mae"])
        for row in loss_log:
            writer.writerow([
                row["epoch"],
                f"{row['train_loss']:.6f}",
                f"{row['val_loss']:.6f}",
                f"{row['train_mae']:.2f}",
                f"{row['val_mae']:.2f}",
            ])
    print(f"  Losses CSV  → {csv_path}")
    return csv_path


def validate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            if outputs.shape != targets.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            loss         = criterion(outputs, targets)
            pred_count   = outputs.sum().item() / 64
            target_count = targets.sum().item() / 64
            total_loss  += loss.item()
            total_mae   += abs(pred_count - target_count)
    return total_loss / len(dataloader), total_mae / len(dataloader)


def plot_density_map(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 2,
                             figsize=(8, num_samples * 3))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    for i in range(min(num_samples, len(dataset))):
        img_tensor, density, _ = dataset[i]
        img_show     = (img_tensor.permute(1,2,0) * std + mean).clamp(0,1).numpy()
        density_show = density.squeeze().numpy()
        count        = density_show.sum() / 64
        axes[i][0].imshow(img_show)
        axes[i][0].set_title(f"Image (count ≈ {count:.0f})", fontsize=8)
        axes[i][0].axis("off")
        axes[i][1].imshow(density_show, cmap="jet")
        axes[i][1].set_title("Ground Truth Density Map", fontsize=8)
        axes[i][1].axis("off")
    plt.suptitle("Real Images + Ground Truth Density Maps",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "real_density_map.png", dpi=150)
    plt.show()
    print(f"[Density Map]  Saved → {out / 'real_density_map.png'}\n")


def train(model, train_loader, val_loader, epochs, model_name="csrnet"):
    model.to(device)
    criterion     = nn.MSELoss()
    optimizer     = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler     = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )
    loss_log      = []
    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")

    print("\n" + "="*60)
    print(f"  TRAINING  –  {model_name.upper()}")
    print(f"  Epochs      : {epochs}")
    print(f"  Image size  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Device      : {device}")
    print(f"  Train size  : {len(train_loader.dataset)}")
    print(f"  Val size    : {len(val_loader.dataset)}")
    print("="*60 + "\n")

    for epoch in range(epochs):
        model.train()
        total_loss  = 0.0
        total_mae   = 0.0
        num_batches = 0

        for images, targets, _ in train_loader:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            if outputs.shape != targets.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            loss         = criterion(outputs, targets)
            pred_count   = outputs.sum().item() / 64
            target_count = targets.sum().item() / 64
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            total_mae   += abs(pred_count - target_count)
            num_batches += 1

        scheduler.step()
        train_loss = total_loss / num_batches
        train_mae  = total_mae  / num_batches
        val_loss, val_mae = validate_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        loss_log.append({
            "epoch"      : epoch + 1,
            "train_loss" : train_loss,
            "val_loss"   : val_loss,
            "train_mae"  : train_mae,
            "val_mae"    : val_mae,
        })

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch+1:>2}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Train MAE: {train_mae:.1f}  "
              f"Val MAE: {val_mae:.1f}  "
              f"LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            best_path = Path(MODEL_DIR) / f"{model_name}_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ Best model saved → {best_path}")

    print("\n  Training completed!")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    final_path = Path(MODEL_DIR) / f"{model_name}_real.pth"
    torch.save(model.state_dict(), final_path)
    print(f"  Final model → {final_path}")
    save_losses_csv(loss_log, OUTPUT_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, epochs + 1)
    axes[0].plot(epochs_range, train_losses, marker="o",
                 color="#7c6af7", lw=2, label="Train Loss")
    axes[0].plot(epochs_range, val_losses, marker="s",
                 color="#f97316", lw=2, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    train_maes = [r["train_mae"] for r in loss_log]
    val_maes   = [r["val_mae"]   for r in loss_log]
    axes[1].plot(epochs_range, train_maes, marker="o",
                 color="#22d3ee", lw=2, label="Train MAE")
    axes[1].plot(epochs_range, val_maes, marker="s",
                 color="#4ade80", lw=2, label="Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (crowd count)")
    axes[1].set_title("Train vs Val MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History  –  {model_name.upper()}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    loss_plot = Path(OUTPUT_DIR) / f"loss_{model_name}.png"
    plt.savefig(loss_plot, dpi=150)
    plt.show()
    print(f"  Loss plot   → {loss_plot}\n")
    return loss_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="csrnet",
                        choices=["csrnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8)
    args = parser.parse_args()

    TRAIN_DIR = (DATA_ROOT.parent.parent /
                 "dataset" / "archive" /
                 "part_A_final" / "train_data")
    VAL_DIR   = (DATA_ROOT.parent.parent /
                 "dataset" / "archive" /
                 "part_B_final" / "test_data")

    print("[ STEP 1 ]  Loading Datasets...")
    train_dataset = ShanghaiTechDataset(TRAIN_DIR,
                                        transform=get_transforms("train"))
    val_dataset   = ShanghaiTechDataset(VAL_DIR,
                                        transform=get_transforms("val"))
    train_loader  = DataLoader(train_dataset, batch_size=args.batch,
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=8,
                               shuffle=False, num_workers=0)

    print("\n[ STEP 2 ]  Plotting Density Maps...")
    plot_density_map(train_dataset, num_samples=4)

    print("[ STEP 3 ]  Building CSRNet...")
    model     = CSRNet()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    print(f"\n[ STEP 4 ]  Training for {args.epochs} epochs...")
    train(model, train_loader, val_loader,
          args.epochs, model_name="csrnet")

    print("="*60)
    print("  ALL DONE!")
    print("  outputs/losses.csv      → all epoch losses")
    print("  outputs/loss_csrnet.png → loss + MAE curves")
    print("  models/csrnet_best.pth  → best model")
    print("  models/csrnet_real.pth  → final model")
    print("="*60 + "\n")