from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from config import CONFIG
from dataset.loader import build_crowd_datasets
from inference.utils import get_device, load_count_calibration, load_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate full metrics/graphs/matrices report for a trained crowd model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-roots", nargs="+", default=[str(CONFIG.project_root / "part_A_final")])
    parser.add_argument("--output-dir", type=str, default=str(CONFIG.output_dir / "best_model_report"))
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=CONFIG.val_split)
    parser.add_argument("--resize-height", type=int, default=CONFIG.image_size[0])
    parser.add_argument("--resize-width", type=int, default=CONFIG.image_size[1])
    parser.add_argument("--calibration-file", type=str, default=None)
    return parser


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = pred - gt
    abs_err = np.abs(err)
    sq_err = err ** 2
    gt_mean = float(np.mean(gt)) if gt.size else 0.0
    sst = float(np.sum((gt - gt_mean) ** 2))
    sse = float(np.sum(sq_err))
    r2 = 1.0 - _safe_div(sse, sst) if sst > 0 else 0.0
    mape = float(np.mean(abs_err / np.clip(gt, a_min=1.0, a_max=None)) * 100.0) if gt.size else 0.0
    medae = float(np.median(abs_err)) if gt.size else 0.0
    pearson = float(np.corrcoef(pred, gt)[0, 1]) if gt.size > 1 else 0.0
    spear = float(spearmanr(pred, gt).correlation) if gt.size > 1 else 0.0
    return {
        "count": float(gt.size),
        "mae": float(np.mean(abs_err)) if gt.size else 0.0,
        "rmse": float(np.sqrt(np.mean(sq_err))) if gt.size else 0.0,
        "mse": float(np.mean(sq_err)) if gt.size else 0.0,
        "bias_mean": float(np.mean(err)) if gt.size else 0.0,
        "bias_median": float(np.median(err)) if gt.size else 0.0,
        "mape_percent": mape,
        "medae": medae,
        "r2": float(r2),
        "pearson_r": pearson,
        "spearman_r": spear,
    }


def build_bins() -> list[tuple[float, float]]:
    return [(0, 100), (100, 300), (300, 700), (700, 1500), (1500, 1e9)]


def bin_label(lo: float, hi: float) -> str:
    hi_txt = "inf" if hi >= 1e9 else str(int(hi))
    return f"{int(lo)}-{hi_txt}"


def per_bin_metrics(pred: np.ndarray, gt: np.ndarray) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    abs_err = np.abs(pred - gt)
    for lo, hi in build_bins():
        mask = (gt >= lo) & (gt < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            out.append({"bin": bin_label(lo, hi), "count": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0})
            continue
        e = pred[mask] - gt[mask]
        out.append(
            {
                "bin": bin_label(lo, hi),
                "count": cnt,
                "mae": float(np.mean(abs_err[mask])),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "bias": float(np.mean(e)),
            }
        )
    return out


def assign_bin(values: np.ndarray) -> np.ndarray:
    bins = build_bins()
    idx = np.zeros(values.shape[0], dtype=np.int64)
    for i, (lo, hi) in enumerate(bins):
        idx[(values >= lo) & (values < hi)] = i
    return idx


def binned_confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = [bin_label(lo, hi) for lo, hi in build_bins()]
    gt_idx = assign_bin(gt)
    pred_idx = assign_bin(pred)
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gt_idx, pred_idx, strict=False):
        mat[g, p] += 1
    return mat, labels


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(gt, pred, s=14, alpha=0.65)
    mx = float(max(np.max(gt), np.max(pred))) if gt.size else 1.0
    plt.plot([0, mx], [0, mx], linestyle="--", linewidth=1.0)
    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_residual_hist(gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    residuals = pred - gt
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=35, alpha=0.85)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Residual (Pred - GT)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_abs_error_vs_gt(gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    abs_err = np.abs(pred - gt)
    plt.figure(figsize=(7, 5))
    plt.scatter(gt, abs_err, s=14, alpha=0.65)
    plt.xlabel("Ground Truth Count")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Crowd Density")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_binned_matrix(matrix: np.ndarray, labels: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Bin")
    ax.set_ylabel("Ground Truth Bin")
    ax.set_title("Binned Count Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def evaluate_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split_dir: Path,
    calibration,
) -> dict[str, object]:
    pred_list: list[float] = []
    gt_list: list[float] = []
    rows: list[dict[str, object]] = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device)
            pred_density = model(images)
            pred_counts = pred_density.flatten(1).sum(1).detach().cpu().numpy().astype(np.float64)
            gt_counts = np.asarray(batch["count"], dtype=np.float64)
            sample_ids = batch["sample_id"]
            image_paths = batch["image_path"]

            for sid, ip, p, g in zip(sample_ids, image_paths, pred_counts.tolist(), gt_counts.tolist(), strict=False):
                pp = float(calibration.apply(p)) if calibration is not None else float(p)
                pred_list.append(pp)
                gt_list.append(float(g))
                rows.append(
                    {
                        "sample_id": sid,
                        "image_path": ip,
                        "gt_count": float(g),
                        "pred_count": pp,
                        "abs_error": float(abs(pp - g)),
                        "residual": float(pp - g),
                    }
                )

    pred = np.asarray(pred_list, dtype=np.float64)
    gt = np.asarray(gt_list, dtype=np.float64)

    split_dir.mkdir(parents=True, exist_ok=True)
    save_csv(split_dir / "predictions.csv", rows)

    metrics = compute_metrics(pred, gt)
    (split_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bins = per_bin_metrics(pred, gt)
    save_csv(split_dir / "bin_metrics.csv", bins)

    quantiles = {
        "abs_error_q50": float(np.quantile(np.abs(pred - gt), 0.50)) if gt.size else 0.0,
        "abs_error_q75": float(np.quantile(np.abs(pred - gt), 0.75)) if gt.size else 0.0,
        "abs_error_q90": float(np.quantile(np.abs(pred - gt), 0.90)) if gt.size else 0.0,
        "abs_error_q95": float(np.quantile(np.abs(pred - gt), 0.95)) if gt.size else 0.0,
    }
    (split_dir / "error_quantiles.json").write_text(json.dumps(quantiles, indent=2), encoding="utf-8")

    mat, labels = binned_confusion_matrix(pred, gt)
    matrix_rows = []
    for i, row in enumerate(mat):
        matrix_rows.append({"gt_bin": labels[i], **{labels[j]: int(v) for j, v in enumerate(row)}})
    save_csv(split_dir / "binned_matrix.csv", matrix_rows)

    plot_scatter(gt, pred, split_dir / "scatter_gt_vs_pred.png")
    plot_residual_hist(gt, pred, split_dir / "residual_histogram.png")
    plot_abs_error_vs_gt(gt, pred, split_dir / "abs_error_vs_gt.png")
    plot_binned_matrix(mat, labels, split_dir / "binned_matrix_heatmap.png")

    top_err_idx = np.argsort(np.abs(pred - gt))[::-1][:20]
    worst_rows = [rows[int(i)] for i in top_err_idx.tolist()]
    save_csv(split_dir / "worst_20_samples.csv", worst_rows)

    return {
        "metrics": metrics,
        "quantiles": quantiles,
        "bin_metrics": bins,
        "num_samples": int(gt.size),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device)
    calibration = load_count_calibration(args.calibration_file)

    _, val_dataset, test_dataset = build_crowd_datasets(
        args.data_roots,
        val_fraction=args.val_fraction,
        resize_to=(args.resize_height, args.resize_width),
        crop_size=None,
        random_flip=False,
        random_crop=False,
        use_cache=True,
        sigma_scale=CONFIG.density_sigma_scale,
        min_sigma=CONFIG.density_min_sigma,
        knn=CONFIG.density_knn,
        seed=CONFIG.random_seed,
    )

    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers}
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    val_report = evaluate_split(model, val_loader, device, out_root / "val", calibration)
    test_report = evaluate_split(model, test_loader, device, out_root / "test", calibration)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration_file": str(Path(args.calibration_file).resolve()) if args.calibration_file else None,
        "val": val_report,
        "test": test_report,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Full Report Complete ===")
    print(f"Report folder: {out_root}")
    print(f"Val MAE={val_report['metrics']['mae']:.3f} RMSE={val_report['metrics']['rmse']:.3f} R2={val_report['metrics']['r2']:.4f}")
    print(f"Test MAE={test_report['metrics']['mae']:.3f} RMSE={test_report['metrics']['rmse']:.3f} R2={test_report['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
