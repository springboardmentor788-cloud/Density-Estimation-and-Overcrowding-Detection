from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from project.inference.utils import FPSMeter, load_model, preprocess_frame, resize_frame_max_dim
from project.utils.visualization import draw_status_banner, make_heatmap, overlay_heatmap


def region_overcrowded(density_map: np.ndarray, region_grid: int, threshold: float) -> bool:
    h, w = density_map.shape
    gh = max(1, h // region_grid)
    gw = max(1, w // region_grid)

    for r in range(region_grid):
        for c in range(region_grid):
            y0 = r * gh
            x0 = c * gw
            y1 = h if r == region_grid - 1 else (r + 1) * gh
            x1 = w if c == region_grid - 1 else (c + 1) * gw
            cell = density_map[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            if float(cell.sum() / cell.size) > threshold:
                return True
    return False


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time crowd monitoring and overcrowding alerts.")
    parser.add_argument("--model", type=str, default="csrnet", choices=["csrnet", "mcnn"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index or video file path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path")

    parser.add_argument("--max-dim", type=int, default=1280)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")

    parser.add_argument("--global-count-threshold", type=float, default=120.0)
    parser.add_argument("--region-grid", type=int, default=4)
    parser.add_argument("--region-density-threshold", type=float, default=0.0025)
    parser.add_argument("--alert-cooldown-sec", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
        use_batch_norm=args.batch_norm,
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_src = fps_src if fps_src > 0 else 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (w, h),
        )

    fps_meter = FPSMeter(window=30)
    last_alert_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame_max_dim(frame, args.max_dim)
        input_tensor = preprocess_frame(frame, device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=args.use_fp16 and device.type == "cuda"):
                pred_density = model(input_tensor)

        density_map = pred_density[0, 0].detach().float().cpu().numpy()
        count = float(density_map.sum())

        heatmap = make_heatmap(density_map, target_size=(frame.shape[1], frame.shape[0]))
        rendered = overlay_heatmap(frame, heatmap, alpha=args.alpha)

        overcrowded = bool(count > args.global_count_threshold) or region_overcrowded(
            density_map,
            region_grid=args.region_grid,
            threshold=args.region_density_threshold,
        )

        fps = fps_meter.tick()
        rendered = draw_status_banner(rendered, count=count, overcrowded=overcrowded, fps=fps)

        now = time.time()
        if overcrowded and (now - last_alert_time) >= args.alert_cooldown_sec:
            print(f"[ALERT] OVERCROWDING detected. Count={count:.2f} Time={time.strftime('%H:%M:%S')}")
            last_alert_time = now

        cv2.imshow("DeepVision Crowd Monitor", rendered)

        if writer is not None:
            out_frame = rendered
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if out_frame.shape[0] != src_h or out_frame.shape[1] != src_w:
                out_frame = cv2.resize(out_frame, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            writer.write(out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
