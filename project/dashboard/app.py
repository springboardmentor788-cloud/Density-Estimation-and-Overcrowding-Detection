from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
import sys
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import streamlit as st

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# Allow launching via `cd dashboard && streamlit run app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


BEST_CHECKPOINT = CONFIG.project_root / "checkpoints" / "high_accuracy_best_partA.pth"
BEST_CALIBRATION = CONFIG.project_root / "outputs" / "count_calibration_partA_push_v1_mae.json"


@dataclass(frozen=True)
class CheckpointChoice:
    label: str
    path: Path
    mae: float
    rmse: float
    epoch: int


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap');
        :root {
          --panel: rgba(10, 22, 38, 0.72);
          --edge: rgba(105, 215, 255, 0.35);
          --ink: #e7f5ff;
          --muted: #99b7c9;
          --accent: #45d3ff;
        }
        .stApp {
          background:
            radial-gradient(1000px 500px at 10% 0%, rgba(49, 129, 255, 0.18), transparent 62%),
            radial-gradient(900px 460px at 88% 16%, rgba(69, 211, 255, 0.14), transparent 60%),
            linear-gradient(140deg, #05101d 0%, #081a2e 45%, #0a1530 100%);
          color: var(--ink);
          font-family: 'Source Sans 3', sans-serif;
        }
        h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.4px; }
        .hero {
          padding: 1.2rem 1.25rem;
          border: 1px solid var(--edge);
          border-radius: 16px;
          background: linear-gradient(130deg, rgba(13, 30, 50, 0.85), rgba(9, 20, 35, 0.82));
          box-shadow: 0 0 0 1px rgba(69, 211, 255, 0.08) inset, 0 16px 46px rgba(0,0,0,.32);
          margin-bottom: 1rem;
        }
        .card {
          border: 1px solid var(--edge);
          border-radius: 14px;
          background: var(--panel);
          padding: 0.8rem 0.95rem;
        }
        .kpi {
          border: 1px solid rgba(69, 211, 255, 0.24);
          border-radius: 12px;
          background: rgba(7, 23, 43, 0.75);
          padding: 0.65rem;
          text-align: center;
        }
        .kpi-label { color: var(--muted); font-size: 0.8rem; }
        .kpi-value { color: var(--accent); font-size: 1.24rem; font-weight: 700; }
        .video-panel {
          border: 1px solid var(--edge);
          border-radius: 16px;
          background: linear-gradient(180deg, rgba(9, 24, 40, 0.82), rgba(5, 16, 29, 0.9));
          padding: 1rem;
          box-shadow: 0 14px 34px rgba(0,0,0,0.28);
        }
        .video-title {
          font-family: 'Orbitron', sans-serif;
          font-size: 1.05rem;
          color: var(--ink);
          margin: 0 0 0.35rem 0;
        }
        .video-note {
          color: var(--muted);
          font-size: 0.88rem;
          margin-bottom: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def discover_checkpoints() -> list[CheckpointChoice]:
    choices: list[CheckpointChoice] = []
    checkpoint_dir = CONFIG.project_root / "checkpoints"
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
            CheckpointChoice(
                label=f"{rel} | MAE {mae:.3f} | RMSE {rmse:.3f} | epoch {epoch}",
                path=path,
                mae=mae,
                rmse=rmse,
                epoch=epoch,
            )
        )

    choices.sort(key=lambda item: (item.mae, item.rmse, item.path.name))
    return choices


@st.cache_resource(show_spinner=False)
def load_runtime(checkpoint_path: str, calibration_path: str | None, prefer_cuda: bool):
    device = get_device(prefer_cuda=prefer_cuda)
    model = load_model(checkpoint_path, device)
    calibration = load_count_calibration(calibration_path) if calibration_path else None
    return model, device, calibration


def process_frame(frame_bgr: np.ndarray, model, device, calibration, threshold: float) -> tuple[np.ndarray, np.ndarray, float, str]:
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


def decode_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def render_image_mode(model, device, calibration, threshold: float, resize_width: int) -> None:
    st.subheader("Image Analyzer")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="img_uploader")
    if uploaded is None:
        st.info("Upload an image to generate overlay and density-map outputs.")
        return

    image = decode_image(uploaded)
    resized = resize_frame_keep_aspect(image, resize_width)
    overlay, density_only, count, status = process_frame(resized, model, device, calibration, threshold)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Overlay | Count {count:.1f} | {status}", use_container_width=True)
    c2.image(cv2.cvtColor(density_only, cv2.COLOR_BGR2RGB), caption="Density Map Only", use_container_width=True)

    out_dir = CONFIG.output_dir / "dashboard" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(uploaded.name).stem
    overlay_path = out_dir / f"{stem}_{stamp}_overlay.jpg"
    density_path = out_dir / f"{stem}_{stamp}_density.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(density_path), density_only)

    st.success(f"Saved outputs: {overlay_path.name}, {density_path.name}")


def _save_uploaded_to_temp(uploaded_file, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _create_video_writer(path: Path, fps: float, size_wh: tuple[int, int]) -> object:
    # Use stable software-friendly codecs for raw export.
    for codec in ("MJPG", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), size_wh)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not initialize video writer with MJPG/XVID/mp4v codecs")


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


def _read_video_bytes(path: Path) -> bytes:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Output video not found or empty: {path}")
    return path.read_bytes()


def run_video_processing(input_path: Path, output_overlay: Path, output_density: Path, model, device, calibration, threshold: float, resize_width: int, sample_fps: float) -> dict[str, float]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(str(input_path))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if src_fps > 0 and sample_fps > 0 and sample_fps < src_fps:
        stride = max(1, int(round(src_fps / sample_fps)))
    out_fps = src_fps / stride if src_fps > 0 else max(1.0, sample_fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    processed = 0
    writer_overlay = None
    writer_density = None
    counts: list[float] = []

    progress = st.progress(0, text="Processing video...")
    status_text = st.empty()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            resized = resize_frame_keep_aspect(frame, resize_width)
            overlay, density_only, count, _ = process_frame(resized, model, device, calibration, threshold)
            counts.append(count)

            if writer_overlay is None:
                h, w = overlay.shape[:2]
                writer_overlay = _create_video_writer(output_overlay, out_fps, (w, h))
                writer_density = _create_video_writer(output_density, out_fps, (w, h))

            writer_overlay.write(overlay)
            writer_density.write(density_only)

            processed += 1
            frame_index += 1
            if total_frames > 0:
                progress.progress(min(1.0, frame_index / total_frames), text=f"Processing frame {frame_index}/{total_frames}")
            status_text.info(f"Frames processed: {processed} | Current count: {count:.1f}")
    finally:
        cap.release()
        if writer_overlay is not None:
            writer_overlay.release()
        if writer_density is not None:
            writer_density.release()

    progress.empty()
    status_text.empty()

    arr = np.asarray(counts, dtype=np.float32)
    return {
        "frames": float(len(counts)),
        "count_mean": float(arr.mean()) if arr.size else 0.0,
        "count_max": float(arr.max()) if arr.size else 0.0,
        "count_min": float(arr.min()) if arr.size else 0.0,
    }


def render_video_mode(model, device, calibration, threshold: float, resize_width: int, checkpoint_choice: CheckpointChoice) -> None:
    st.subheader("Video Analyzer")
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")
    sample_fps = st.slider("Inference FPS sampling", min_value=1.0, max_value=30.0, value=8.0, step=1.0)

    if uploaded is None:
        st.info("Upload a video to generate both overlay and density-only videos.")
        return

    if st.button("Generate Output Videos", type="primary"):
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        temp_input = _save_uploaded_to_temp(uploaded, suffix=suffix)

        out_dir = CONFIG.output_dir / "dashboard" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(uploaded.name).stem

        overlay_raw_path = out_dir / f"{stem}_{stamp}_overlay_raw.avi"
        density_raw_path = out_dir / f"{stem}_{stamp}_density_raw.avi"
        overlay_path = out_dir / f"{stem}_{stamp}_overlay.mp4"
        density_path = out_dir / f"{stem}_{stamp}_density.mp4"

        stats = run_video_processing(
            temp_input,
            overlay_raw_path,
            density_raw_path,
            model,
            device,
            calibration,
            threshold,
            resize_width,
            sample_fps,
        )

        final_overlay = _transcode_to_web_mp4(overlay_raw_path, overlay_path)
        final_density = _transcode_to_web_mp4(density_raw_path, density_path)

        st.success("Video processing complete")

        overlay_bytes = _read_video_bytes(final_overlay)
        density_bytes = _read_video_bytes(final_density)

        stat_cols = st.columns(5)
        stat_cols[0].markdown(f"<div class='kpi'><div class='kpi-label'>Frames</div><div class='kpi-value'>{int(stats['frames'])}</div></div>", unsafe_allow_html=True)
        stat_cols[1].markdown(f"<div class='kpi'><div class='kpi-label'>Avg Count</div><div class='kpi-value'>{stats['count_mean']:.1f}</div></div>", unsafe_allow_html=True)
        stat_cols[2].markdown(f"<div class='kpi'><div class='kpi-label'>Max Count</div><div class='kpi-value'>{stats['count_max']:.1f}</div></div>", unsafe_allow_html=True)
        stat_cols[3].markdown(f"<div class='kpi'><div class='kpi-label'>Min Count</div><div class='kpi-value'>{stats['count_min']:.1f}</div></div>", unsafe_allow_html=True)
        stat_cols[4].markdown(f"<div class='kpi'><div class='kpi-label'>Model MAE</div><div class='kpi-value'>{checkpoint_choice.mae:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='card' style='margin-top:0.8rem;'><b>Selected model</b>: {checkpoint_choice.label}<br/>"
            f"<b>Threshold</b>: {threshold:.1f} | <b>Sampling FPS</b>: {sample_fps:.1f} | <b>Source</b>: {uploaded.name}</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            st.markdown("<div class='video-panel'><div class='video-title'>Overlay Video</div><div class='video-note'>Frame + density heatmap + count/status HUD.</div></div>", unsafe_allow_html=True)
            st.video(overlay_bytes, format="video/mp4")
            st.download_button("Download Overlay Video", data=overlay_bytes, file_name=final_overlay.name, mime="video/mp4")
        with right:
            st.markdown("<div class='video-panel'><div class='video-title'>Density Map Video</div><div class='video-note'>Heatmap-only visualization for intensity analysis.</div></div>", unsafe_allow_html=True)
            st.video(density_bytes, format="video/mp4")
            st.download_button("Download Density Video", data=density_bytes, file_name=final_density.name, mime="video/mp4")


def render_live_mode(model, device, calibration, threshold: float, resize_width: int) -> None:
    st.subheader("Live Camera Density Monitor")
    st.caption("Runs local webcam inference and streams both overlay and density-only panels.")

    duration_sec = st.slider("Run duration (seconds)", min_value=5, max_value=120, value=20, step=5)
    target_fps = st.slider("Preview FPS", min_value=2, max_value=20, value=10, step=1)

    if st.button("Start Live Monitor", type="primary"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam device 0.")
            return

        slot_left, slot_right = st.columns(2)
        left_panel = slot_left.empty()
        right_panel = slot_right.empty()
        metric_slot = st.empty()

        start_t = time.time()
        next_tick = start_t
        counts: list[float] = []

        try:
            while (time.time() - start_t) < duration_sec:
                ok, frame = cap.read()
                if not ok:
                    break
                resized = resize_frame_keep_aspect(frame, resize_width)
                overlay, density_only, count, status = process_frame(resized, model, device, calibration, threshold)
                counts.append(count)

                left_panel.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Overlay | {status} | {count:.1f}", use_container_width=True)
                right_panel.image(cv2.cvtColor(density_only, cv2.COLOR_BGR2RGB), caption="Density Map", use_container_width=True)

                metric_slot.markdown(
                    f"<div class='card'>Live stats: current={count:.1f}, mean={np.mean(counts):.1f}, max={np.max(counts):.1f}, frames={len(counts)}</div>",
                    unsafe_allow_html=True,
                )

                next_tick += 1.0 / float(target_fps)
                sleep_for = max(0.0, next_tick - time.time())
                time.sleep(sleep_for)
        finally:
            cap.release()

        st.success("Live monitor session complete")


def main() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for dashboard mode")

    st.set_page_config(page_title="DeepVision Crowd Monitor", page_icon="DV", layout="wide")
    apply_theme()

    st.markdown(
        """
        <div class="hero">
          <h1>DeepVision Crowd Monitor Dashboard</h1>
          <p style="margin:0;color:#bcd9ea;">High-accuracy CSRNet operations console for live monitoring and media analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Runtime")
        checkpoint_choices = discover_checkpoints()
        if checkpoint_choices:
            default_index = 0
            for idx, choice in enumerate(checkpoint_choices):
                if choice.path == BEST_CHECKPOINT:
                    default_index = idx
                    break
            selected_choice = st.selectbox(
                "Trained model",
                options=checkpoint_choices,
                index=default_index,
                format_func=lambda choice: choice.label,
            )
            checkpoint_path = str(selected_choice.path)
        else:
            selected_choice = CheckpointChoice(label=f"{BEST_CHECKPOINT.name} (fallback)", path=BEST_CHECKPOINT, mae=float("nan"), rmse=float("nan"), epoch=-1)
            checkpoint_path = str(BEST_CHECKPOINT)
            st.text_input("Checkpoint", value=checkpoint_path)

        calibration_path = st.text_input("Calibration (optional)", value=str(BEST_CALIBRATION if BEST_CALIBRATION.exists() else ""))
        threshold = st.slider("Overcrowding threshold", min_value=20, max_value=1000, value=130, step=5)
        resize_width = st.slider("Inference resize width", min_value=320, max_value=1920, value=1280, step=32)
        prefer_cuda = st.checkbox("Use CUDA if available", value=True)

        if not Path(checkpoint_path).exists():
            st.error("Checkpoint file not found.")
            st.stop()
        if calibration_path and not Path(calibration_path).exists():
            st.warning("Calibration file not found. Raw counts will be used.")

    model, device, calibration = load_runtime(
        checkpoint_path,
        calibration_path if calibration_path and Path(calibration_path).exists() else None,
        prefer_cuda,
    )

    st.markdown(
        f"<div class='card'>Model loaded on <b>{device}</b>. Selected checkpoint: <b>{selected_choice.label}</b>."
        f" Dashboard outputs include both overlay and density-only views.</div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Live Footage", "Upload Image", "Upload Video"])

    with tabs[0]:
        render_live_mode(model, device, calibration, threshold, resize_width)
    with tabs[1]:
        render_image_mode(model, device, calibration, threshold, resize_width)
    with tabs[2]:
        render_video_mode(model, device, calibration, threshold, resize_width, selected_choice)


if __name__ == "__main__":
    main()
