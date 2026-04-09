from __future__ import annotations

import argparse
from pathlib import Path
import time

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from config import CONFIG
from inference.utils import CrowdAlertState, annotate_frame, get_device, load_count_calibration, load_model, overlay_density, predict_frame, resize_frame_keep_aspect


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video crowd counting inference")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=CONFIG.alert_threshold)
    parser.add_argument("--resize-width", type=int, default=CONFIG.inference_resize_width)
    parser.add_argument("--sample-fps", type=float, default=CONFIG.video_sample_fps)
    parser.add_argument("--sleep-ms", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--calibration-file", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    CONFIG.ensure_dirs()
    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device)
    calibration = load_count_calibration(args.calibration_file)
    alert_state = CrowdAlertState(threshold=args.threshold, cooldown_frames=CONFIG.alert_cooldown_frames)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(args.input)

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if source_fps > 0.0 and args.sample_fps > 0.0 and args.sample_fps < source_fps:
        stride = max(1, int(round(source_fps / args.sample_fps)))
    output_fps = source_fps / stride if source_fps > 0.0 else args.sample_fps

    writer = None
    frame_index = 0
    processed_index = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            resized = resize_frame_keep_aspect(frame, args.resize_width)
            density, count = predict_frame(model, resized, device)
            if calibration is not None:
                count = calibration.apply(count)
            if args.debug and processed_index == 0:
                print(f"DEBUG video | frame={tuple(frame.shape)} resized={tuple(resized.shape)} density={tuple(density.shape)} count={count:.2f}")
            status, should_alert = alert_state.update(count, processed_index)
            if should_alert:
                print(f"ALERT frame={processed_index} count={count:.1f} threshold={args.threshold:.1f}")

            overlay = overlay_density(resized, density)
            overlay = annotate_frame(overlay, count, status, alert=should_alert)

            if writer is None:
                height, width = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, max(output_fps, 1.0), (width, height))
            writer.write(overlay)

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

            processed_index += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
