from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
except Exception:  # pragma: no cover
    torch = None
    DataLoader = object
    ConcatDataset = object
    WeightedRandomSampler = object

from config import CONFIG
from dataset.density_map import resize_density_tensor
from dataset.loader import build_crowd_datasets
from inference.utils import get_device
from models.csrnet import CSRNet
from training.validate import evaluate_model
from utils.weight_loader import load_external_weights
from dataset.density_map import load_points_from_mat_cached


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CSRNet for crowd counting")
    parser.add_argument("--data-roots", nargs="+", default=[str(CONFIG.project_root / "part_A_final")])
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=CONFIG.val_split)
    parser.add_argument("--resize-height", type=int, default=CONFIG.image_size[0])
    parser.add_argument("--resize-width", type=int, default=CONFIG.image_size[1])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CONFIG.checkpoint_dir))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-weights", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-random-crop", action="store_true")
    parser.add_argument("--crop-height", type=int, default=CONFIG.crop_size[0] if CONFIG.crop_size else 256)
    parser.add_argument("--crop-width", type=int, default=CONFIG.crop_size[1] if CONFIG.crop_size else 256)
    parser.add_argument("--save-examples-dir", type=str, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-7)
    parser.add_argument("--count-loss-weight", type=float, default=CONFIG.count_loss_weight)
    parser.add_argument("--count-loss-weight-max", type=float, default=CONFIG.count_loss_weight_max)
    parser.add_argument("--dense-oversample", action="store_true")
    parser.add_argument("--dense-threshold", type=float, default=700.0)
    parser.add_argument("--dense-weight", type=float, default=4.0)
    return parser


def _sample_count_from_item(item: object) -> float:
    mat_path = item.mat_path if hasattr(item, "mat_path") else None
    if mat_path is None:
        return 0.0
    pts = load_points_from_mat_cached(str(mat_path))
    return float(pts.shape[0])


def _build_sample_weights(train_dataset: object, threshold: float, dense_weight: float) -> list[float]:
    weights: list[float] = []
    if isinstance(train_dataset, ConcatDataset):
        for sub_ds in train_dataset.datasets:
            sub_items = getattr(sub_ds, "samples", [])
            for item in sub_items:
                cnt = _sample_count_from_item(item)
                weights.append(dense_weight if cnt >= threshold else 1.0)
    else:
        items = getattr(train_dataset, "samples", [])
        for item in items:
            cnt = _sample_count_from_item(item)
            weights.append(dense_weight if cnt >= threshold else 1.0)
    if not weights:
        weights = [1.0] * len(train_dataset)
    return weights


def train_one_epoch(
    model: CSRNet,
    dataloader: DataLoader,
    device: "torch.device",
    optimizer: "torch.optim.Optimizer",
    scaler: "torch.cuda.amp.GradScaler | None",
    *,
    grad_clip_norm: float,
    use_amp: bool,
    count_loss_weight: float,
) -> float:
    criterion = torch.nn.MSELoss()
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)
        gt_count = batch["count"].to(device, non_blocking=True).float() if torch.is_tensor(batch["count"]) else torch.tensor(batch["count"], device=device).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_density = model(images)
            target_density = resize_density_tensor(gt_density, pred_density.shape[-2:])
            density_loss = criterion(pred_density, target_density)
            pred_count = pred_density.flatten(1).sum(dim=1)
            count_loss = criterion(pred_count, gt_count)
            loss = density_loss + count_loss_weight * count_loss

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.item())
        valid_batches += 1

    return running_loss / max(1, valid_batches)


def save_checkpoint(
    path: str | Path,
    model: CSRNet,
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    metrics: dict[str, float],
    *,
    scheduler: "torch.optim.lr_scheduler.ReduceLROnPlateau | None" = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "image_size": CONFIG.image_size,
            "learning_rate": CONFIG.learning_rate,
            "batch_size": CONFIG.batch_size,
        },
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = build_arg_parser().parse_args()
    CONFIG.ensure_dirs()

    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_dataset, val_dataset, _ = build_crowd_datasets(
        args.data_roots,
        val_fraction=args.val_fraction,
        resize_to=(args.resize_height, args.resize_width),
        crop_size=(args.crop_height, args.crop_width) if args.train_random_crop else None,
        random_flip=True,
        random_crop=args.train_random_crop,
        use_cache=True,
        sigma_scale=CONFIG.density_sigma_scale,
        min_sigma=CONFIG.density_min_sigma,
        knn=CONFIG.density_knn,
        seed=CONFIG.random_seed,
        max_samples=args.max_samples,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_sampler = None
    if args.dense_oversample:
        sample_weights = _build_sample_weights(train_dataset, args.dense_threshold, args.dense_weight)
        train_sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = CSRNet(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    use_amp = CONFIG.use_amp and (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_mae = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_mae = float(checkpoint.get("metrics", {}).get("mae", best_mae))
    elif args.init_weights is not None:
        missing, unexpected = load_external_weights(model, args.init_weights, map_location=device)
        print(
            f"Initialized from external weights: {args.init_weights} | missing={len(missing)} | unexpected={len(unexpected)}"
        )

    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "csrnet_best.pth"
    last_path = checkpoint_dir / "csrnet_last.pth"

    for epoch in range(start_epoch, args.epochs + 1):
        if args.epochs <= start_epoch:
            count_loss_weight = args.count_loss_weight_max
        else:
            ratio = (epoch - start_epoch) / max(1, (args.epochs - start_epoch))
            count_loss_weight = args.count_loss_weight + ratio * (args.count_loss_weight_max - args.count_loss_weight)
        train_loss = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            grad_clip_norm=CONFIG.grad_clip_norm,
            use_amp=use_amp,
            count_loss_weight=count_loss_weight,
        )
        metrics = evaluate_model(model, val_loader, device, save_examples_dir=args.save_examples_dir if epoch == args.epochs else None)
        metrics["train_loss"] = train_loss
        metrics["count_loss_weight"] = float(count_loss_weight)

        scheduler.step(metrics["mae"])
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        save_checkpoint(last_path, model, optimizer, epoch, metrics, scheduler=scheduler)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            save_checkpoint(best_path, model, optimizer, epoch, metrics, scheduler=scheduler)

        print(
            f"Epoch {epoch:03d} | Train Loss {train_loss:.6f} | Val Loss {metrics['val_loss']:.6f} | MAE {metrics['mae']:.3f} | RMSE {metrics['rmse']:.3f} | Bias {metrics['bias_mean']:.2f} | LR {metrics['lr']:.2e}"
        )

    print(f"Best validation MAE: {best_mae:.3f}")


if __name__ == "__main__":
    main()
