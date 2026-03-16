# ============================================================
#  scripts/predict.py
#  Run:  python scripts/predict.py
#  Run:  python scripts/predict.py --image path\to\image.jpg
# ============================================================

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_DIR, OUTPUT_DIR, DATA_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

IMG_SIZE = 224


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


def predict(model, image_path):
    """
    Predicts crowd count using Test Time Augmentation (TTA).
    Runs 3 predictions and averages for better accuracy.
    """
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    flip_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    bright_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    img  = Image.open(image_path).convert("RGB")
    w, h = img.size

    model.eval()
    counts = []
    with torch.no_grad():
        for t in [base_transform, flip_transform, bright_transform]:
            tensor = t(img).unsqueeze(0).to(device)
            output = model(tensor)
            count  = output.sum().item() / 64
            counts.append(count)

    pred_count = np.mean(counts)

    base_tensor = base_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        density_out = model(base_tensor)
    density_map = density_out.squeeze().cpu().numpy()

    print(f"\n  Image           : {Path(image_path).name}")
    print(f"  Original size   : {w} x {h}")
    print(f"  Prediction 1    : {counts[0]:.0f} people (original)")
    print(f"  Prediction 2    : {counts[1]:.0f} people (flipped)")
    print(f"  Prediction 3    : {counts[2]:.0f} people (brighter)")
    print(f"  Final (avg TTA) : {pred_count:.0f} people ✅")

    mean     = torch.tensor([0.485, 0.456, 0.406])
    std      = torch.tensor([0.229, 0.224, 0.225])
    img_show = base_tensor.squeeze().permute(1,2,0).cpu()
    img_show = (img_show * std + mean).clamp(0,1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_show)
    axes[0].set_title(f"Input Image\n{Path(image_path).name}",
                      fontsize=9)
    axes[0].axis("off")

    im = axes[1].imshow(density_map, cmap="jet")
    axes[1].set_title(
        f"Predicted Density Map\nCrowd Count ≈ {pred_count:.0f} people",
        fontsize=9
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.suptitle(
        f"Crowd Prediction  –  {pred_count:.0f} people detected",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"prediction_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved → {save_path}\n")
    return pred_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file")
    args = parser.parse_args()

    if args.image is None:
        sample = (DATA_ROOT.parent.parent /
                  "dataset" / "archive" /
                  "part_A_final" / "test_data" /
                  "images" / "IMG_1.jpg")
        args.image = str(sample)
        print(f"  No image given — using sample:\n  {args.image}")

    print("[ STEP 1 ]  Loading Best Model...")
    model      = CSRNet().to(device)
    model_path = Path(MODEL_DIR) / "csrnet_best.pth"
    if not model_path.exists():
        model_path = Path(MODEL_DIR) / "csrnet_real.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded → {model_path}")

    print("\n[ STEP 2 ]  Predicting with TTA...")
    count = predict(model, args.image)

    print("="*50)
    print(f"  Predicted crowd count : {count:.0f} people")
    print("="*50 + "\n")