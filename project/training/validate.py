from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.utils.metrics import RunningMetrics
from project.utils.visualization import save_gt_pred_comparison


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    vis_dir: Path | None = None,
    max_visualizations: int = 5,
    return_records: bool = False,
) -> dict:
    model.eval()

    running_loss = 0.0
    metric = RunningMetrics()
    vis_count = 0
    records = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)
        gt_count = batch["count"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred_density = model(images)
            loss = criterion(pred_density, gt_density)

        pred_count = pred_density.flatten(1).sum(dim=1)
        metric.update(pred_count, gt_count)
        running_loss += loss.item() * images.size(0)

        if return_records:
            for i in range(images.shape[0]):
                pred_val = float(pred_count[i].item())
                gt_val = float(gt_count[i].item())
                err = pred_val - gt_val
                records.append(
                    {
                        "image_path": str(batch["image_path"][i]),
                        "gt_count": gt_val,
                        "pred_count": pred_val,
                        "abs_error": abs(err),
                        "sq_error": err * err,
                    }
                )

        if vis_dir is not None and vis_count < max_visualizations:
            bsz = images.shape[0]
            for i in range(bsz):
                if vis_count >= max_visualizations:
                    break
                name = Path(batch["image_path"][i]).stem
                save_gt_pred_comparison(
                    gt_density=gt_density[i, 0].detach().cpu().numpy(),
                    pred_density=pred_density[i, 0].detach().cpu().numpy(),
                    out_path=vis_dir / f"{name}_gt_vs_pred.png",
                    title=f"{name} | GT {gt_count[i].item():.1f} | Pred {pred_count[i].item():.1f}",
                    image_path=str(batch["image_path"][i]),
                    gt_count=float(gt_count[i].item()),
                    pred_count=float(pred_count[i].item()),
                )
                vis_count += 1

    n_samples = len(loader.dataset)
    result = {
        "loss": running_loss / max(1, n_samples),
        "mae": metric.mae,
        "rmse": metric.rmse,
        "r2": metric.r2,
    }
    if return_records:
        result["records"] = records
    return result
