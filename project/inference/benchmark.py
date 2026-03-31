from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from project.inference.utils import load_model, preprocess_frame, resize_frame_max_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark crowd model inference FPS.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["csrnet", "mcnn"], default="csrnet")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-dim", type=int, default=1280)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--fp16", action="store_true")
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

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    image_bgr = resize_frame_max_dim(image_bgr, args.max_dim)
    inp = preprocess_frame(image_bgr, device)

    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                _ = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(args.iters):
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                _ = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    fps = args.iters / max(elapsed, 1e-8)
    print(f"FPS: {fps:.3f}")
    print(f"Latency(ms): {1000.0 / max(fps, 1e-8):.3f}")


if __name__ == "__main__":
    main()
