from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from project.dataset.loader import create_dataloader
from project.models.csrnet import CSRNet
from project.models.mcnn import MCNN
from project.training.validate import validate_epoch
from project.utils.visualization import save_metrics_panel, save_regression_plots


def build_model(name: str, use_bn: bool) -> torch.nn.Module:
    if name.lower() == "csrnet":
        return CSRNet(load_pretrained_frontend=False, use_batch_norm=use_bn)
    if name.lower() == "mcnn":
        return MCNN()
    raise ValueError(f"Unsupported model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crowd counting model on ShanghaiTech split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--part", type=str, choices=["A", "B"], default="A")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default="csrnet", choices=["csrnet", "mcnn"])
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--output-stride", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("project/data/cache"))
    parser.add_argument("--vis-dir", type=Path, default=Path("project/data/eval_vis"))
    parser.add_argument("--report-dir", type=Path, default=Path("project/data/eval_reports"))
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def write_per_image_csv(records: list[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "gt_count", "pred_count", "abs_error", "sq_error"],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_dataloader(
        dataset_root=args.dataset_root,
        part=args.part,
        split=args.split,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        max_dim=args.max_dim,
        crop_size=args.crop_size,
        output_stride=args.output_stride,
        train=False,
    )

    model = build_model(args.model, args.batch_norm).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    criterion = nn.MSELoss(reduction="mean")
    metrics = validate_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_amp=args.use_amp,
        vis_dir=args.vis_dir,
        return_records=True,
    )

    records = metrics.pop("records", [])
    report_name = f"{args.model}_part{args.part}_{args.split}_{args.checkpoint.stem}"
    report_root = args.report_dir / report_name
    report_root.mkdir(parents=True, exist_ok=True)

    per_image_csv = report_root / "per_image_metrics.csv"
    summary_json = report_root / "summary.json"
    metrics_panel = report_root / "metrics_panel.png"

    write_per_image_csv(records, per_image_csv)

    gt_counts = [r["gt_count"] for r in records]
    pred_counts = [r["pred_count"] for r in records]
    if gt_counts and pred_counts:
        save_regression_plots(
            gt_counts=gt_counts,
            pred_counts=pred_counts,
            out_dir=report_root,
            prefix="counts",
        )
    save_metrics_panel(metrics, metrics_panel)

    summary = {
        "dataset_root": str(args.dataset_root),
        "part": args.part,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "metrics": metrics,
        "num_samples": len(records),
        "artifacts": {
            "per_image_csv": str(per_image_csv),
            "metrics_panel": str(metrics_panel),
            "scatter_plot": str(report_root / "counts_scatter_gt_vs_pred.png"),
            "residual_hist": str(report_root / "counts_residual_hist.png"),
            "evaluation_matrix": str(report_root / "counts_evaluation_matrix.png"),
            "density_visualizations_dir": str(args.vis_dir),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation complete")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Report directory: {report_root}")


if __name__ == "__main__":
    main()
