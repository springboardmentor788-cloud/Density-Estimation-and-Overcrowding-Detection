from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from config import CONFIG
from inference.utils import annotate_frame, get_device, load_count_calibration, load_model, overlay_density, predict_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crowd counting on a single image or folder")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(CONFIG.output_dir / "image_inference"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=CONFIG.alert_threshold)
    parser.add_argument("--resize-width", type=int, default=CONFIG.inference_resize_width)
    parser.add_argument("--calibration-file", type=str, default=None)
    return parser


def infer_image(model, image_path: Path, output_dir: Path, device, threshold: float, resize_width: int, calibration) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    from inference.utils import resize_frame_keep_aspect

    resized = resize_frame_keep_aspect(image, resize_width)
    density, count = predict_frame(model, resized, device)
    if calibration is not None:
        count = calibration.apply(count)
    status = "OVERCROWDED" if count >= threshold else "SAFE"
    overlay = overlay_density(resized, density)
    overlay = annotate_frame(overlay, count, status, alert=status == "OVERCROWDED")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_overlay.jpg"
    cv2.imwrite(str(output_path), overlay)
    print(f"{image_path.name}: count={count:.1f} status={status} -> {output_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    CONFIG.ensure_dirs()
    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device)
    calibration = load_count_calibration(args.calibration_file)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_dir():
        for image_path in sorted(input_path.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            infer_image(model, image_path, output_dir, device, args.threshold, args.resize_width, calibration)
    else:
        infer_image(model, input_path, output_dir, device, args.threshold, args.resize_width, calibration)


if __name__ == "__main__":
    main()
