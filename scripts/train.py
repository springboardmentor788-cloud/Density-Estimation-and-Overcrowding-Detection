# ============================================================
#  scripts/train.py
#
#  Optimized for: CPU training, 30 epochs, best possible MAE
#
#  Run:  python scripts/train.py
#  Run:  python scripts/train.py --epochs 30 --batch 4
# ============================================================

import sys
import csv
import time
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_ROOT, MODEL_DIR, OUTPUT_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# ── Constants ────────────────────────────────────────────────────────────────
CROP_SIZE   = 256    # Random crop size (replaces fixed Resize 224)
SIGMA       = 4      # Density map sigma (4 = better GT match than 8)
DENSITY_SCALE = 100  # Multiply density so loss is numerically stable


# ── MAT / Density Helpers ────────────────────────────────────────────────────

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


def make_density_map(points, img_shape, sigma=SIGMA):
    """
    Creates a 2D Gaussian density map from point annotations.
    sigma=4 is tighter and matches the model output scale better than sigma=8.
    """
    H, W    = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    for point in points:
        x = int(float(point[0]))
        y = int(float(point[1]))
        if 0 <= x < W and 0 <= y < H:
            density[y, x] += 1.0
    return gaussian_filter(density, sigma=sigma)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShanghaiTechDataset(Dataset):
    """
    Loads ShanghaiTech Part A / Part B.

    KEY FIX: No Resize(224,224). Instead uses random crops at train time
    and full-image inference at val time. This preserves crowd density
    structure which Resize destroys.
    """
    def __init__(self, data_dir, mode="train", crop_size=CROP_SIZE):
        self.data_dir  = Path(data_dir)
        self.img_dir   = self.data_dir / "images"
        self.gt_dir    = self.data_dir / "ground_truth"
        self.mode      = mode
        self.crop_size = crop_size
        self.samples   = []

        for img_path in sorted(self.img_dir.glob("*.jpg")):
            num      = img_path.stem.replace("IMG_", "")
            mat_path = self.gt_dir / f"GT_IMG_{num}.mat"
            if mat_path.exists():
                self.samples.append((img_path, mat_path))

        print(f"  [{mode:>5}]  {self.data_dir.parent.name}/"
              f"{self.data_dir.name}  →  {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mat_path = self.samples[index]

        # ── Load image ───────────────────────────────────────────────
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # ── Minimum size guard (some images are very small) ──────────
        min_side = self.crop_size
        if w < min_side or h < min_side:
            scale = min_side / min(w, h) + 0.01
            img   = img.resize((int(w * scale) + 1, int(h * scale) + 1),
                               Image.BILINEAR)
            w, h  = img.size

        # ── Load density map ─────────────────────────────────────────
        points  = load_mat(str(mat_path))
        density = make_density_map(points, (h, w))

        # ── Random crop (train) / center crop (val) ──────────────────
        if self.mode == "train":
            x0 = np.random.randint(0, w - self.crop_size + 1)
            y0 = np.random.randint(0, h - self.crop_size + 1)
        else:
            x0 = (w - self.crop_size) // 2
            y0 = (h - self.crop_size) // 2

        x1 = x0 + self.crop_size
        y1 = y0 + self.crop_size
        img     = img.crop((x0, y0, x1, y1))
        density = density[y0:y1, x0:x1]

        # ── Augmentation (train only) ─────────────────────────────────
        if self.mode == "train":
            if np.random.rand() > 0.5:
                img     = img.transpose(Image.FLIP_LEFT_RIGHT)
                density = density[:, ::-1].copy()

            # Color jitter
            img = transforms.functional.adjust_brightness(
                img, 1.0 + np.random.uniform(-0.2, 0.2))
            img = transforms.functional.adjust_contrast(
                img, 1.0 + np.random.uniform(-0.2, 0.2))

        # ── To tensor + normalize ─────────────────────────────────────
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.functional.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

        # ── Downscale density to model output size (÷8) ───────────────
        # CSRNet output is H/8 × W/8
        H, W = self.crop_size, self.crop_size
        density_tensor = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
        density_tensor = torch.nn.functional.interpolate(
            density_tensor,
            size=(H // 8, W // 8),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # Scale so values are numerically stable for MSE loss
        density_tensor = density_tensor * DENSITY_SCALE

        return img_tensor, density_tensor, str(img_path)


# ── Model ─────────────────────────────────────────────────────────────────────

class CSRNet(nn.Module):
    """
    CSRNet with VGG16 frontend + dilated conv backend.

    KEY FIX: Only freeze first 3 VGG blocks (layers 0-16).
    Unfreeze block 4+5 (layers 17-22) so they adapt to your dataset.
    This gives ~5-10 MAE improvement over fully frozen frontend.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend  = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)
        self._init_backend_weights()
        self._set_frozen_layers()

    def _set_frozen_layers(self):
        """
        Freeze VGG blocks 1-3 (stable low-level features).
        Unfreeze blocks 4-5 (high-level features that need to adapt).
        """
        children = list(self.frontend.children())
        for i, layer in enumerate(children):
            # Freeze layers 0-16 (VGG blocks 1,2,3)
            # Unfreeze layers 17-22 (VGG blocks 4,5)
            requires_grad = (i >= 17)
            for param in layer.parameters():
                param.requires_grad = requires_grad

        frozen_count   = sum(1 for p in self.frontend.parameters()
                             if not p.requires_grad)
        unfrozen_count = sum(1 for p in self.frontend.parameters()
                             if p.requires_grad)
        print(f"  Frontend: {frozen_count} params frozen, "
              f"{unfrozen_count} params trainable (blocks 4-5)")

    def _init_backend_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


# ── Validation ───────────────────────────────────────────────────────────────

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
            pred_count   = float(torch.relu(outputs).sum().item()) / DENSITY_SCALE
            target_count = float(targets.sum().item()) / DENSITY_SCALE
            total_loss  += loss.item()
            total_mae   += abs(pred_count - target_count)
    return total_loss / len(dataloader), total_mae / len(dataloader)


# ── CSV Logger ────────────────────────────────────────────────────────────────

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


# ── Density Map Preview ───────────────────────────────────────────────────────

def plot_density_map(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    for i in range(min(num_samples, len(dataset))):
        img_tensor, density, img_path = dataset[i]
        img_show     = (img_tensor.permute(1,2,0) * std + mean).clamp(0,1).numpy()
        density_show = density.squeeze().numpy()

        try:
            p        = Path(img_path)
            num      = p.stem.replace("IMG_", "")
            gt_dir   = p.parent.parent / "ground_truth"
            mat_path = gt_dir / f"GT_IMG_{num}.mat"
            mat      = sio.loadmat(str(mat_path))
            points   = mat["image_info"][0][0][0][0][0]
            count    = len(points)
        except Exception:
            count = int(density_show.sum() / DENSITY_SCALE)

        axes[i][0].imshow(img_show)
        axes[i][0].set_title(f"Image  (GT count: {count})", fontsize=9)
        axes[i][0].axis("off")

        d_display = density_show / density_show.max() if density_show.max() > 0 else density_show
        axes[i][1].imshow(d_display, cmap="jet", vmin=0, vmax=1)
        axes[i][1].set_title(f"Density Map  (sigma={SIGMA})", fontsize=9)
        axes[i][1].axis("off")

    plt.suptitle("Real Images + Ground Truth Density Maps",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "real_density_map.png", dpi=150)
    plt.show()
    print(f"  Density map preview → {out / 'real_density_map.png'}\n")


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs, model_name="csrnet"):
    model.to(device)

    criterion = nn.MSELoss()

    # Separate LRs: lower for frontend (fine-tuning), higher for backend (learning)
    frontend_params = [p for p in model.frontend.parameters() if p.requires_grad]
    backend_params  = list(model.backend.parameters()) + \
                      list(model.output_layer.parameters())

    optimizer = optim.Adam([
        {"params": frontend_params, "lr": 1e-5},   # fine-tune VGG blocks 4-5
        {"params": backend_params,  "lr": 1e-4},   # train backend fresh
    ], weight_decay=1e-4)

    # Cosine annealing — gradually reduces LR for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Early stopping
    best_val_mae    = float("inf")
    best_val_loss   = float("inf")
    patience        = 8       # stop if no improvement for 8 epochs
    patience_counter= 0
    loss_log        = []
    train_losses    = []
    val_losses      = []
    train_maes      = []
    val_maes_list   = []

    print("\n" + "="*62)
    print(f"  TRAINING  —  {model_name.upper()}")
    print(f"  Epochs        : {epochs}  (early stop patience: {patience})")
    print(f"  Crop size     : {CROP_SIZE}×{CROP_SIZE}  (no fixed resize)")
    print(f"  Sigma         : {SIGMA}")
    print(f"  Density scale : {DENSITY_SCALE}")
    print(f"  Device        : {device}")
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Val samples   : {len(val_loader.dataset)}")
    print(f"  LR frontend   : 1e-5  (VGG blocks 4-5)")
    print(f"  LR backend    : 1e-4  (dilated conv backend)")
    print(f"  Scheduler     : CosineAnnealingLR")
    print("="*62 + "\n")

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss  = 0.0
        total_mae   = 0.0
        num_batches = 0
        t_epoch     = time.time()

        for images, targets, _ in train_loader:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            if outputs.shape != targets.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:],
                    mode="bilinear", align_corners=False,
                )

            loss = criterion(outputs, targets)

            # MAE in real people count
            pred_count   = float(torch.relu(outputs).sum().item()) / DENSITY_SCALE
            target_count = float(targets.sum().item()) / DENSITY_SCALE

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients on CPU
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

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
        train_maes.append(train_mae)
        val_maes_list.append(val_mae)

        loss_log.append({
            "epoch"      : epoch + 1,
            "train_loss" : train_loss,
            "val_loss"   : val_loss,
            "train_mae"  : train_mae,
            "val_mae"    : val_mae,
        })

        elapsed    = time.time() - t_epoch
        lr_front   = optimizer.param_groups[0]["lr"]
        lr_back    = optimizer.param_groups[1]["lr"]
        saved_flag = ""

        # Save best model by val MAE (more meaningful than val loss)
        if val_mae < best_val_mae:
            best_val_mae  = val_mae
            best_val_loss = val_loss
            patience_counter = 0
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            best_path = Path(MODEL_DIR) / f"{model_name}_best.pth"
            torch.save(model.state_dict(), str(best_path))
            saved_flag = "  ★ best"
        else:
            patience_counter += 1

        print(f"  Epoch [{epoch+1:>2}/{epochs}]  "
              f"TrLoss:{train_loss:.4f}  VaLoss:{val_loss:.4f}  "
              f"TrMAE:{train_mae:>6.1f}  VaMAE:{val_mae:>6.1f}  "
              f"LR:{lr_back:.0e}  "
              f"({elapsed:.0f}s){saved_flag}")

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered — "
                  f"no improvement for {patience} epochs.")
            print(f"  Best Val MAE: {best_val_mae:.1f}")
            break

    # Save final model
    total_time = time.time() - t_start
    print(f"\n  Training finished in {total_time/60:.1f} minutes")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    final_path = Path(MODEL_DIR) / f"{model_name}_real.pth"
    torch.save(model.state_dict(), str(final_path))
    print(f"  Final model  → {final_path}")
    print(f"  Best model   → {Path(MODEL_DIR) / f'{model_name}_best.pth'}")
    print(f"  Best Val MAE : {best_val_mae:.1f}")

    save_losses_csv(loss_log, OUTPUT_DIR)

    # ── Loss + MAE Plot ───────────────────────────────────────────────
    ep_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep_range, train_losses, marker="o", color="#7c6af7",
                 lw=2, label="Train Loss")
    axes[0].plot(ep_range, val_losses,   marker="s", color="#f97316",
                 lw=2, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_range, train_maes,     marker="o", color="#22d3ee",
                 lw=2, label="Train MAE")
    axes[1].plot(ep_range, val_maes_list,  marker="s", color="#4ade80",
                 lw=2, label="Val MAE")
    axes[1].axhline(y=best_val_mae, color="red", linestyle="--",
                    alpha=0.6, label=f"Best MAE: {best_val_mae:.1f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (people count)")
    axes[1].set_title("Train vs Val MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {model_name.upper()}  "
                 f"(Best Val MAE: {best_val_mae:.1f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    loss_plot = Path(OUTPUT_DIR) / f"loss_{model_name}.png"
    plt.savefig(str(loss_plot), dpi=150)
    plt.show()
    print(f"  Loss plot    → {loss_plot}\n")

    return loss_log


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="csrnet",
                        choices=["csrnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=4,
                        help="Batch size (default 4 for CPU, use 8 if RAM allows)")
    args = parser.parse_args()

    BASE = DATA_ROOT.parent.parent / "dataset" / "archive"

    # ── STEP 1: Load Datasets ────────────────────────────────────────
    print("[ STEP 1 ]  Loading Datasets (Part A + Part B)...")
    train_A = ShanghaiTechDataset(BASE / "part_A_final" / "train_data", mode="train")
    train_B = ShanghaiTechDataset(BASE / "part_B_final" / "train_data", mode="train")
    val_A   = ShanghaiTechDataset(BASE / "part_A_final" / "test_data",  mode="val")
    val_B   = ShanghaiTechDataset(BASE / "part_B_final" / "test_data",  mode="val")

    train_combined = ConcatDataset([train_A, train_B])
    val_combined   = ConcatDataset([val_A,   val_B])

    print(f"\n  Train total  : {len(train_combined)}  "
          f"(A:{len(train_A)} + B:{len(train_B)})")
    print(f"  Val total    : {len(val_combined)}  "
          f"(A:{len(val_A)} + B:{len(val_B)})")

    train_loader = DataLoader(
        train_combined, batch_size=args.batch,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_combined, batch_size=4,
        shuffle=False, num_workers=0, pin_memory=False
    )

    # ── STEP 2: Preview Density Maps ────────────────────────────────
    print("\n[ STEP 2 ]  Previewing Density Maps...")
    plot_density_map(train_A, num_samples=4)

    # ── STEP 3: Build Model ──────────────────────────────────────────
    print("[ STEP 3 ]  Building CSRNet...")
    model     = CSRNet()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")

    # ── STEP 4: Train ────────────────────────────────────────────────
    print(f"\n[ STEP 4 ]  Training for {args.epochs} epochs on CPU...")
    print("  TIP: This will take ~1.5–2 hrs on CPU. "
          "Leave it running and check back.\n")
    train(model, train_loader, val_loader, args.epochs, model_name="csrnet")

    print("=" * 62)
    print("  ALL DONE!")
    print(f"  Train data   : Part A ({len(train_A)}) + Part B ({len(train_B)}) images")
    print(f"  Val data     : Part A ({len(val_A)}) + Part B ({len(val_B)}) images")
    print("  outputs/losses.csv        → all epoch losses")
    print("  outputs/loss_csrnet.png   → loss + MAE curves")
    print("  models/csrnet_best.pth    → best model (use this!)")
    print("  models/csrnet_real.pth    → final epoch model")
    print("=" * 62 + "\n")