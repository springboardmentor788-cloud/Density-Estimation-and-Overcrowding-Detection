from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from project.dataset.loader import create_dataloader
from project.models.csrnet import CSRNet
from project.models.mcnn import MCNN
from project.training.validate import validate_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, use_batch_norm: bool) -> torch.nn.Module:
    if name.lower() == "csrnet":
        return CSRNet(load_pretrained_frontend=True, use_batch_norm=use_batch_norm)
    if name.lower() == "mcnn":
        return MCNN()
    raise ValueError(f"Unsupported model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSRNet/MCNN for crowd counting.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--part", choices=["A", "B"], default="A")

    parser.add_argument("--model", choices=["csrnet", "mcnn"], default="csrnet")
    parser.add_argument("--batch-norm", action="store_true")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--output-stride", type=int, default=8)

    parser.add_argument("--cache-dir", type=Path, default=Path("project/data/cache"))
    parser.add_argument("--work-dir", type=Path, default=Path("project/data/runs"))
    parser.add_argument("--exp-name", type=str, default="csrnet_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def save_training_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["val_mae"], label="val_mae")
    plt.plot(history["epoch"], history["val_rmse"], label="val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("Count Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["val_r2"], label="val_r2")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "r2_curve.png", dpi=160)
    plt.close()


def append_csv(log_file: Path, row: dict) -> None:
    write_header = not log_file.exists()
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.work_dir / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    vis_dir = exp_dir / "val_vis"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        dataset_root=args.dataset_root,
        part=args.part,
        split="train",
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        max_dim=args.max_dim,
        crop_size=args.crop_size,
        output_stride=args.output_stride,
        train=True,
    )
    val_loader = create_dataloader(
        dataset_root=args.dataset_root,
        part=args.part,
        split="test",
        cache_root=args.cache_dir,
        batch_size=max(1, args.batch_size // 2),
        workers=max(1, args.workers // 2),
        max_dim=args.max_dim,
        crop_size=args.crop_size,
        output_stride=args.output_stride,
        train=False,
    )

    model = build_model(args.model, args.batch_norm).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_mae = float("inf")

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_mae = float(ckpt.get("best_mae", best_mae))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
    }
    log_file = exp_dir / "metrics.csv"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            gt_density = batch["density"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred_density = model(images)
                loss = criterion(pred_density, gt_density)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = epoch_loss / max(1, len(train_loader.dataset))

        val_metrics = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
            vis_dir=vis_dir if (epoch + 1) % 10 == 0 else None,
            max_visualizations=6,
        )

        scheduler.step(val_metrics["mae"])

        is_best = val_metrics["mae"] < best_mae
        best_mae = min(best_mae, val_metrics["mae"])

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae": best_mae,
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
        }

        torch.save(ckpt, ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "best_mae": best_mae,
        }
        append_csv(log_file, row)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])

        print(
            f"Epoch {epoch + 1:03d} | "
            f"TrainLoss {train_loss:.6f} | "
            f"ValLoss {val_metrics['loss']:.6f} | "
            f"MAE {val_metrics['mae']:.4f} | "
            f"RMSE {val_metrics['rmse']:.4f} | "
            f"R2 {val_metrics['r2']:.4f} | "
            f"BestMAE {best_mae:.4f}"
        )

    save_training_curves(history, exp_dir)
    print(f"Training complete. Artifacts at: {exp_dir}")


if __name__ == "__main__":
    main()
