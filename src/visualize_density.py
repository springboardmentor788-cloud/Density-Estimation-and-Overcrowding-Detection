"""
Visualize density maps in 2x3 layout: Crowd Density Estimation Results
with metrics (Predicted, Ground Truth, Error) and MAE on error map.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import PROJECT_ROOT
from csrnet import CSRNet, density_to_count
from dataset import ShanghaiTechDensityDataset


def normalize_density_for_display(d: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Normalize density so the color range spreads properly (avoid mostly one color)."""
    d = np.asarray(d, dtype=np.float32)
    d = np.clip(d, 0, None)
    if d.size == 0 or d.max() <= 0:
        return d
    p = np.percentile(d, percentile)
    if p <= 0:
        p = d.max()
    out = (d / p).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def density_to_rgb(density_vis: np.ndarray) -> np.ndarray:
    """
    Map normalized density [0,1] to RGB: blue (low) -> green (mid) -> red (high).
    Returns (H, W, 3) float in [0, 1].
    """
    d = np.clip(np.asarray(density_vis, dtype=np.float32), 0.0, 1.0)
    r = np.where(d < 0.5, 0.0, 2.0 * (d - 0.5))
    g = np.where(d < 0.5, 2.0 * d, 1.0)
    b = np.where(d < 0.5, 1.0 - 2.0 * d, 0.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def overlay_density_on_image(
    image: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.45,
    mask_style: bool = True,
    use_jet: bool = True,
    percentile: float = 98.0,
) -> np.ndarray:
    """
    Overlay density on image as a thin, semi-transparent mask.
    use_jet: True = jet colormap (blue-cyan-green-yellow-red); else RGB gradient.
    percentile: normalize by this percentile so colors spread properly.
    """
    h, w = image.shape[:2]
    density_resized = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    density_resized = np.clip(density_resized, 0, None).astype(np.float32)
    density_vis = normalize_density_for_display(density_resized, percentile=percentile)

    # Soft edges: light blur so mask looks like smooth spots
    density_vis = cv2.GaussianBlur(density_vis, (0, 0), sigmaX=2.0)
    if density_vis.max() > 1e-8:
        density_vis = np.clip(density_vis / density_vis.max(), 0, 1).astype(np.float32)

    if use_jet:
        density_color = plt.cm.jet(density_vis)[:, :, :3].astype(np.float32)
    else:
        density_color = density_to_rgb(density_vis)

    if mask_style:
        mask = (np.power(density_vis, 0.6) * alpha).clip(0, 1)
        mask_3 = mask[:, :, np.newaxis]
        overlay = (1 - mask_3) * image + mask_3 * density_color
    else:
        overlay = (alpha * density_color + (1 - alpha) * image).clip(0, 1)

    return overlay.astype(np.float32).clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="A")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_calibrate", action="store_true", help="Do not calibrate predicted scale to match GT")
    parser.add_argument("--calibrate_samples", type=int, default=30, help="Samples to use for calibration scale")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ShanghaiTechDensityDataset(part=args.part, split=args.split)
    loader = DataLoader(dataset, batch_size=max(4, args.num), shuffle=True)

    model = CSRNet(pretrained=False)
    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.is_absolute():
            cwd_path = p.resolve()
            ckpt_path = cwd_path if cwd_path.is_file() else (PROJECT_ROOT / p).resolve()
        else:
            ckpt_path = p
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Train first: cd src && python train.py --part A --epochs 50"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    imgs, gt_densities, gt_counts = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        pred_densities = model(imgs)

    # Calibration: scale predicted density so mean predicted count matches mean GT count
    calibrate_scale = 1.0
    if not args.no_calibrate and args.calibrate_samples > 0:
        sum_gt, sum_pred, num_s = 0.0, 0.0, 0
        cal_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        with torch.no_grad():
            for im, gt, _ in cal_loader:
                if num_s >= args.calibrate_samples:
                    break
                im = im.to(device)
                pred = model(im)
                pred_np = pred.clamp(min=0).cpu().numpy()
                gt_np = gt.numpy()
                for i in range(pred_np.shape[0]):
                    if num_s >= args.calibrate_samples:
                        break
                    sum_gt += float(np.clip(gt_np[i], 0, None).sum())
                    sum_pred += float(np.clip(pred_np[i], 0, None).sum())
                    num_s += 1
        if sum_pred > 1e-8 and num_s > 0:
            calibrate_scale = sum_gt / sum_pred
            print(f"Calibration: scale = {calibrate_scale:.3f} (so predicted count matches GT scale)")

    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_disp = (imgs.cpu() * std + mean).clamp(0, 1)
    imgs_np = imgs_disp.permute(0, 2, 3, 1).numpy()

    n = min(args.num, imgs.size(0))

    for idx in range(n):
        img = imgs_np[idx]
        gt_d = gt_densities[idx, 0].numpy()
        pred_d = (pred_densities[idx, 0].cpu().numpy() * calibrate_scale).astype(np.float32)
        # Counts: sum of density map (same source as displayed map); ensure non-negative
        gt_count = float(np.clip(gt_d, 0, None).sum())
        pred_count = float(np.clip(pred_d, 0, None).sum())
        count_error = abs(pred_count - gt_count)
        # Use same shapes for error (resize GT to pred if needed)
        if pred_d.shape != gt_d.shape:
            gt_d_for_err = cv2.resize(gt_d, (pred_d.shape[1], pred_d.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            gt_d_for_err = gt_d
        density_mae = float(np.mean(np.abs(pred_d.astype(np.float64) - gt_d_for_err.astype(np.float64))))

        # 2x3 layout like reference
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Title and metrics
        fig.suptitle("Crowd Density Estimation Results", fontsize=14, fontweight="bold", y=1.02)
        fig.text(0.5, 0.98, f"Predicted: {pred_count:.1f}, Ground Truth: {gt_count:.1f}, Error: {count_error:.1f}", ha="center", fontsize=11)

        # Row 0: Original | Ground Truth Density | Predicted Density
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        gt_d_norm = normalize_density_for_display(gt_d, percentile=98.0)
        axes[0, 1].imshow(plt.cm.jet(gt_d_norm)[:, :, :3])
        axes[0, 1].set_title(f"Ground Truth Density\nCount: {gt_count:.1f}")
        axes[0, 1].axis("off")

        pred_d_norm = normalize_density_for_display(pred_d, percentile=98.0)
        axes[0, 2].imshow(plt.cm.jet(pred_d_norm)[:, :, :3])
        axes[0, 2].set_title(f"Predicted Density\nCount: {pred_count:.1f}")
        axes[0, 2].axis("off")

        # Row 1: GT Overlay | Pred Overlay | Absolute Error Map (jet colormap for visibility)
        gt_overlay = overlay_density_on_image(img, gt_d, alpha=0.5, mask_style=True, use_jet=True, percentile=98.0)
        axes[1, 0].imshow(gt_overlay)
        axes[1, 0].set_title("GT Density Overlay")
        axes[1, 0].axis("off")

        pred_overlay = overlay_density_on_image(img, pred_d, alpha=0.5, mask_style=True, use_jet=True, percentile=98.0)
        axes[1, 1].imshow(pred_overlay)
        axes[1, 1].set_title("Predicted Density Overlay")
        axes[1, 1].axis("off")

        # Absolute error map: |predicted density - ground truth density| (same shape, same scale)
        abs_err = np.abs(pred_d.astype(np.float64) - gt_d_for_err.astype(np.float64))
        # Use percentile for color scale so spatial variation is visible (not dominated by one peak)
        vmax = float(np.percentile(abs_err, 99)) if abs_err.size > 0 else 1.0
        if vmax < 1e-10:
            vmax = 1.0
        im_err = axes[1, 2].imshow(abs_err, cmap="hot", vmin=0, vmax=vmax)
        axes[1, 2].set_title(f"Absolute Error Map |Pred − GT|\nDensity MAE: {density_mae:.4f}, Count error: {count_error:.1f}")
        axes[1, 2].axis("off")
        cbar = plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)
        cbar.set_label("|pred − GT|", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_name = "density_vis.png" if idx == 0 else f"density_vis_{idx + 1}.png"
        out_path = PROJECT_ROOT / out_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved {out_path}")

    print(f"Done. First figure saved as density_vis.png in project root.")


if __name__ == "__main__":
    main()
