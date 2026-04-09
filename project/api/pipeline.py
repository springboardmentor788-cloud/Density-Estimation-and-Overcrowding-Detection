from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
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
from inference.utils import (
    annotate_frame,
    density_to_heatmap,
    get_device,
    load_count_calibration,
    load_model,
    overlay_density,
    predict_frame,
    resize_frame_keep_aspect,
)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    label: str
    path: str
    mae: float
    rmse: float
    epoch: int


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the API backend")


def discover_models() -> list[ModelChoice]:
    checkpoint_dir = CONFIG.project_root / "checkpoints"
    choices: list[ModelChoice] = []
    if not checkpoint_dir.exists():
        return choices

    for path in sorted(checkpoint_dir.rglob("*.pth")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {}) or {}
        mae = float(metrics.get("mae", float("inf")))
        rmse = float(metrics.get("rmse", float("inf")))
        epoch = int(payload.get("epoch", -1))
        rel = path.relative_to(CONFIG.project_root)
        choices.append(
            ModelChoice(
                label=f"{rel} | MAE {mae:.3f} | RMSE {rmse:.3f} | epoch {epoch}",
                path=str(path),
                mae=mae,
                rmse=rmse,
                epoch=epoch,
            )
        )

    choices.sort(key=lambda item: (item.mae, item.rmse, item.path))
    return choices


@lru_cache(maxsize=8)
def load_runtime(checkpoint_path: str, calibration_path: str | None, prefer_cuda: bool = True):
    device = get_device(prefer_cuda=prefer_cuda)
    model = load_model(checkpoint_path, device)
    calibration = load_count_calibration(calibration_path) if calibration_path else None
    return model, device, calibration


def _decode_image(image_bytes: bytes) -> np.ndarray:
    _ensure_cv2()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def _encode_image(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    _ensure_cv2()
    ok, buffer = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError("Could not encode image")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _transcode_to_web_mp4(input_path: Path, output_path: Path) -> Path:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return input_path

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        return input_path
    return output_path


def _create_video_writer(path: Path, fps: float, size_wh: tuple[int, int]) -> object:
    _ensure_cv2()
    for codec in ("MJPG", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), size_wh)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not initialize video writer with MJPG/XVID/mp4v codecs")


def _process_frame(frame_bgr: np.ndarray, model, device, calibration, threshold: float) -> tuple[np.ndarray, np.ndarray, float, str]:
    density, count = predict_frame(model, frame_bgr, device)
    if calibration is not None:
        count = calibration.apply(count)
    status = "OVERCROWDED" if count >= threshold else "SAFE"
    overlay = annotate_frame(overlay_density(frame_bgr, density), count, status, alert=status == "OVERCROWDED")
    density_only = density_to_heatmap(density)
    if density_only.shape[:2] != frame_bgr.shape[:2]:
        density_only = cv2.resize(density_only, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    density_only = annotate_frame(density_only, count, status, alert=status == "OVERCROWDED")
    return overlay, density_only, float(count), status


def infer_image_bytes(
    image_bytes: bytes,
    *,
    checkpoint_path: str,
    calibration_path: str | None,
    threshold: float,
    resize_width: int,
    prefer_cuda: bool = True,
) -> dict[str, object]:
    _ensure_cv2()
    model, device, calibration = load_runtime(checkpoint_path, calibration_path, prefer_cuda)
    image = _decode_image(image_bytes)
    resized = resize_frame_keep_aspect(image, resize_width)
    overlay, density_only, count, status = _process_frame(resized, model, device, calibration, threshold)
    return {
        "count": count,
        "status": status,
        "overlay_image": _encode_image(overlay, ".jpg"),
        "density_image": _encode_image(density_only, ".jpg"),
    }


def infer_video_file(
    input_path: Path,
    *,
    checkpoint_path: str,
    calibration_path: str | None,
    threshold: float,
    resize_width: int,
    sample_fps: float,
    output_dir: Path,
    prefer_cuda: bool = True,
) -> dict[str, object]:
    _ensure_cv2()
    model, device, calibration = load_runtime(checkpoint_path, calibration_path, prefer_cuda)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(str(input_path))

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if source_fps > 0 and sample_fps > 0 and sample_fps < source_fps:
        stride = max(1, int(round(source_fps / sample_fps)))
    output_fps = source_fps / stride if source_fps > 0 else max(1.0, sample_fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_raw_path = output_dir / "overlay_raw.avi"
    density_raw_path = output_dir / "density_raw.avi"
    overlay_mp4_path = output_dir / "overlay.mp4"
    density_mp4_path = output_dir / "density.mp4"

    writer_overlay = None
    writer_density = None
    frame_index = 0
    processed = 0
    counts: list[float] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            resized = resize_frame_keep_aspect(frame, resize_width)
            overlay, density_only, count, _ = _process_frame(resized, model, device, calibration, threshold)
            counts.append(count)

            if writer_overlay is None:
                height, width = overlay.shape[:2]
                writer_overlay = _create_video_writer(overlay_raw_path, output_fps, (width, height))
                writer_density = _create_video_writer(density_raw_path, output_fps, (width, height))

            writer_overlay.write(overlay)
            writer_density.write(density_only)

            processed += 1
            frame_index += 1
    finally:
        capture.release()
        if writer_overlay is not None:
            writer_overlay.release()
        if writer_density is not None:
            writer_density.release()

    final_overlay = _transcode_to_web_mp4(overlay_raw_path, overlay_mp4_path)
    final_density = _transcode_to_web_mp4(density_raw_path, density_mp4_path)

    arr = np.asarray(counts, dtype=np.float32)
    return {
        "frames": float(len(counts)),
        "count_mean": float(arr.mean()) if arr.size else 0.0,
        "count_max": float(arr.max()) if arr.size else 0.0,
        "count_min": float(arr.min()) if arr.size else 0.0,
        "overlay_path": str(final_overlay),
        "density_path": str(final_density),
    }
