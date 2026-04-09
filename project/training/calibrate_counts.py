from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import minimize

from config import CONFIG
from dataset.loader import build_crowd_datasets
from inference.utils import get_device
from models.csrnet import CSRNet
from utils.metrics import mae, rmse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit linear count calibration pred->gt")
    parser.add_argument("--data-roots", nargs="+", default=[str(CONFIG.project_root / "part_A_final")])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(CONFIG.output_dir / "count_calibration.json"))
    parser.add_argument("--resize-height", type=int, default=CONFIG.image_size[0])
    parser.add_argument("--resize-width", type=int, default=CONFIG.image_size[1])
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=CONFIG.val_split)
    parser.add_argument("--objective", type=str, choices=["mae", "rmse"], default="mae")
    return parser


def _fit_l2(pred_np: np.ndarray, gt_np: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred_np, gt_np, deg=1)
    return float(slope), float(intercept)


def _fit_mae(pred_np: np.ndarray, gt_np: np.ndarray) -> tuple[float, float]:
    def objective(params: np.ndarray) -> float:
        s, b = float(params[0]), float(params[1])
        cal = np.clip(s * pred_np + b, a_min=0.0, a_max=None)
        return float(np.mean(np.abs(cal - gt_np)))

    s0, b0 = _fit_l2(pred_np, gt_np)
    res = minimize(objective, x0=np.array([s0, b0], dtype=np.float64), method="Nelder-Mead")
    if not res.success:
        return s0, b0
    return float(res.x[0]), float(res.x[1])


def main() -> None:
    args = build_arg_parser().parse_args()
    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)

    _, val_dataset, _ = build_crowd_datasets(
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
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CSRNet(pretrained=False)
    model.load_weights(args.checkpoint, map_location=device)
    model.to(device).eval()

    preds, gts = [], []
    with torch.inference_mode():
        for batch in dataloader:
            x = batch["image"].to(device)
            c = model(x).flatten(1).sum(1).float().cpu().numpy()
            y = batch["count"].float().cpu().numpy()
            preds.extend(c.tolist())
            gts.extend(y.tolist())

    pred_np = np.asarray(preds, dtype=np.float64)
    gt_np = np.asarray(gts, dtype=np.float64)
    if args.objective == "rmse":
        slope, intercept = _fit_l2(pred_np, gt_np)
    else:
        slope, intercept = _fit_mae(pred_np, gt_np)
    calibrated = np.clip(slope * pred_np + intercept, a_min=0.0, a_max=None)

    payload = {
        "slope": float(slope),
        "intercept": float(intercept),
        "raw_mae": float(mae(pred_np.tolist(), gt_np.tolist())),
        "raw_rmse": float(rmse(pred_np.tolist(), gt_np.tolist())),
        "calibrated_mae": float(mae(calibrated.tolist(), gt_np.tolist())),
        "calibrated_rmse": float(rmse(calibrated.tolist(), gt_np.tolist())),
        "objective": args.objective,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved calibration to {out}")


if __name__ == "__main__":
    main()
