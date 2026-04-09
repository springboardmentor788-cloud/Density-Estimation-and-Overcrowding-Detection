from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = object

from config import CONFIG
from dataset.density_map import resize_density_tensor
from dataset.loader import build_crowd_datasets
from inference.utils import get_device
from models.csrnet import CSRNet
from utils.metrics import summarize_counts
from utils.visualization import save_comparison_figure


def denormalize_tensor(image_tensor: "torch.Tensor") -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.asarray(CONFIG.image_std, dtype=np.float32) + np.asarray(CONFIG.image_mean, dtype=np.float32)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image[..., ::-1]


def evaluate_model(
    model: CSRNet,
    dataloader: "DataLoader",
    device: "torch.device",
    *,
    save_examples_dir: str | Path | None = None,
    max_examples: int = 4,
    debug: bool = False,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for validation")
    criterion = torch.nn.MSELoss()
    predictions: list[float] = []
    targets: list[float] = []
    losses: list[float] = []
    saved_examples = 0

    model.eval()
    with torch.no_grad():
        printed_debug = False
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            gt_density = batch["density"].to(device, non_blocking=True)
            gt_counts = batch["count"].cpu().numpy().tolist() if torch.is_tensor(batch["count"]) else list(batch["count"])

            pred_density = model(images)
            resized_gt = resize_density_tensor(gt_density, pred_density.shape[-2:])
            density_loss = criterion(pred_density, resized_gt)
            pred_counts = pred_density.flatten(1).sum(dim=1)
            gt_count_tensor = batch["count"].to(device).float() if torch.is_tensor(batch["count"]) else torch.tensor(batch["count"], device=device).float()
            count_loss = criterion(pred_counts, gt_count_tensor)
            loss = density_loss + CONFIG.count_loss_weight * count_loss
            losses.append(float(loss.item()))

            if debug and not printed_debug:
                print(
                    f"DEBUG shapes | image={tuple(images.shape)} gt={tuple(gt_density.shape)} pred={tuple(pred_density.shape)} | gt_sum={float(gt_density.sum()):.2f} pred_sum={float(pred_density.sum()):.2f}"
                )
                printed_debug = True

            batch_pred_counts = pred_density.flatten(1).sum(dim=1).detach().cpu().numpy().tolist()
            predictions.extend(batch_pred_counts)
            targets.extend([float(value) for value in gt_counts])

            if save_examples_dir is not None and saved_examples < max_examples:
                save_root = Path(save_examples_dir)
                save_root.mkdir(parents=True, exist_ok=True)
                for sample_index in range(images.shape[0]):
                    if saved_examples >= max_examples:
                        break
                    image_bgr = denormalize_tensor(images[sample_index])
                    gt_resized = resized_gt[sample_index, 0].detach().cpu().numpy()
                    pred_map = pred_density[sample_index, 0].detach().cpu().numpy()
                    out_path = save_root / f"example_{saved_examples + 1:02d}.jpg"
                    save_comparison_figure(
                        image_bgr,
                        gt_resized,
                        pred_map,
                        gt_count=float(gt_counts[sample_index]),
                        pred_count=float(batch_pred_counts[sample_index]),
                        output_path=out_path,
                    )
                    saved_examples += 1

    metrics = summarize_counts(predictions, targets)
    pred_np = np.asarray(predictions, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    err = pred_np - tgt_np
    abs_err = np.abs(err)
    metrics["bias_mean"] = float(np.mean(err)) if err.size else 0.0

    bins = [(0, 100), (100, 300), (300, 700), (700, 1000000)]
    for lo, hi in bins:
        mask = (tgt_np >= lo) & (tgt_np < hi)
        key = f"mae_bin_{lo}_{hi}"
        metrics[key] = float(np.mean(abs_err[mask])) if np.any(mask) else 0.0

    metrics["val_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CSRNet crowd counting")
    parser.add_argument("--data-roots", nargs="+", default=[str(CONFIG.project_root / "part_A_final")])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--resize-height", type=int, default=CONFIG.image_size[0])
    parser.add_argument("--resize-width", type=int, default=CONFIG.image_size[1])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--val-fraction", type=float, default=CONFIG.val_split)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-examples-dir", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    CONFIG.ensure_dirs()
    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)

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
        max_samples=args.max_samples,
    )

    dataset = val_dataset if len(val_dataset) > 0 else test_dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CSRNet(pretrained=False)
    model.load_weights(args.checkpoint, map_location=device)
    model.to(device)

    metrics = evaluate_model(
        model,
        dataloader,
        device,
        save_examples_dir=args.save_examples_dir,
        max_examples=args.max_examples,
        debug=args.debug,
    )
    print(
        f"Validation | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | Loss: {metrics['val_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
