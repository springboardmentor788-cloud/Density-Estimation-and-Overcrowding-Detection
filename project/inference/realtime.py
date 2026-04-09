from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Real-time webcam crowd counting")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=CONFIG.alert_threshold)
    parser.add_argument("--resize-width", type=int, default=CONFIG.inference_resize_width)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--calibration-file", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    CONFIG.ensure_dirs()
    device = get_device(prefer_cuda=args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device)
    calibration = load_count_calibration(args.calibration_file)
    alert_state = CrowdAlertState(threshold=args.threshold, cooldown_frames=CONFIG.alert_cooldown_frames)

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(args.source)

    processed_index = 0
    frame_index = 0
    last_overlay = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % max(1, args.frame_stride) == 0:
                resized = resize_frame_keep_aspect(frame, args.resize_width)
                density, count = predict_frame(model, resized, device)
                if calibration is not None:
                    count = calibration.apply(count)
                status, should_alert = alert_state.update(count, processed_index)
                if should_alert:
                    print(f"ALERT frame={processed_index} count={count:.1f} threshold={args.threshold:.1f}")
                overlay = overlay_density(resized, density)
                last_overlay = annotate_frame(overlay, count, status, alert=should_alert)
                processed_index += 1

            display = last_overlay if last_overlay is not None else frame
            cv2.imshow("DeepVision Crowd Monitor", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
