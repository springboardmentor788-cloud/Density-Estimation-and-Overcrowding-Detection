from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    denom = x.max() + 1e-8
    x /= denom
    return x


def make_heatmap(density_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    density = normalize_map(density_map)
    density_u8 = (density * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    return heatmap


def overlay_heatmap(frame_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def save_gt_pred_comparison(
    gt_density: np.ndarray,
    pred_density: np.ndarray,
    out_path: Path,
    title: str,
    image_path: str | None = None,
    gt_count: float | None = None,
    pred_count: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[12, 3], hspace=0.18)

    # Original image panel.
    ax0 = fig.add_subplot(gs[0, 0])
    if image_path is not None:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax0.imshow(img_rgb)
        else:
            ax0.text(0.5, 0.5, "Original image\nnot found", ha="center", va="center", fontsize=11)
    else:
        ax0.text(0.5, 0.5, "Original image\nnot provided", ha="center", va="center", fontsize=11)
    ax0.set_title("Original Image", fontsize=12, fontweight="bold")
    ax0.axis("off")

    # GT density panel.
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(gt_density, cmap="jet")
    ax1.set_title("Ground Truth Density", fontsize=12, fontweight="bold")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Predicted density panel.
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(pred_density, cmap="jet")
    ax2.set_title("Predicted Density", fontsize=12, fontweight="bold")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Detailed metrics strip.
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    if gt_count is not None and pred_count is not None:
        err = float(pred_count - gt_count)
        mae = abs(err)
        mse = err * err
        detail = (
            f"GT Count: {gt_count:.2f}    "
            f"Predicted Count: {pred_count:.2f}    "
            f"Error (Pred-GT): {err:.2f}    "
            f"MAE: {mae:.2f}    "
            f"MSE: {mse:.2f}"
        )
    else:
        detail = "Count statistics unavailable"
    ax3.text(0.01, 0.76, title, fontsize=12, fontweight="bold", va="top")
    ax3.text(0.01, 0.26, detail, fontsize=11, family="monospace", va="top")

    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_status_banner(frame: np.ndarray, count: float, overcrowded: bool, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    status_text = "OVERCROWDED" if overcrowded else "SAFE"
    status_color = (0, 0, 255) if overcrowded else (0, 180, 0)

    cv2.rectangle(frame, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.putText(frame, f"Count: {count:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status_text}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if overcrowded:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 4)

    return frame


def save_regression_plots(
    gt_counts: np.ndarray,
    pred_counts: np.ndarray,
    out_dir: Path,
    prefix: str = "eval",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_counts = np.asarray(gt_counts, dtype=np.float32)
    pred_counts = np.asarray(pred_counts, dtype=np.float32)
    residuals = pred_counts - gt_counts

    # GT vs prediction scatter.
    max_val = float(max(np.max(gt_counts), np.max(pred_counts), 1.0))
    plt.figure(figsize=(6, 6))
    plt.scatter(gt_counts, pred_counts, s=12, alpha=0.65)
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.2)
    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("GT vs Prediction")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_scatter_gt_vs_pred.png", dpi=180)
    plt.close()

    # Residual histogram.
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=30, alpha=0.8, edgecolor="black")
    plt.xlabel("Residual (Pred - GT)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_residual_hist.png", dpi=180)
    plt.close()

    # Error matrix heatmap (2D histogram of GT and prediction bins).
    bins = 16
    hist, xedges, yedges = np.histogram2d(gt_counts, pred_counts, bins=bins)
    plt.figure(figsize=(6, 5))
    plt.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    plt.colorbar(label="Samples")
    plt.xlabel("GT Count")
    plt.ylabel("Pred Count")
    plt.title("Evaluation Matrix (2D Count Histogram)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_evaluation_matrix.png", dpi=180)
    plt.close()


def save_metrics_panel(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.axis("off")

    lines = [
        f"MAE:  {metrics.get('mae', float('nan')):.4f}",
        f"RMSE: {metrics.get('rmse', float('nan')):.4f}",
        f"R2:   {metrics.get('r2', float('nan')):.4f}",
        f"Loss: {metrics.get('loss', float('nan')):.6f}",
    ]
    ax.text(0.02, 0.92, "Evaluation Summary", fontsize=13, fontweight="bold", va="top")
    ax.text(0.02, 0.72, "\n".join(lines), fontsize=11, va="top", family="monospace")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
