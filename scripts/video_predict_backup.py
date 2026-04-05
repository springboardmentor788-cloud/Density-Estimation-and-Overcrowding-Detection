# ============================================================
#  scripts/video_predict.py
#
#  Standalone video crowd counting pipeline using CSRNet.
#
#  Usage:
#    python scripts/video_predict.py --video path/to/video.mp4
#    python scripts/video_predict.py --video path/to/video.mp4 --show-density
#    python scripts/video_predict.py --video path/to/video.mp4 --save
#    python scripts/video_predict.py --video path/to/video.mp4 --model best --skip 3
# ============================================================

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_DIR, OUTPUT_DIR, DATA_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")


# ── Model Definitions ────────────────────────────────────────────────────────

class CSRNet(nn.Module):
    """3-layer backend: 256->128->64  (csrnet_best.pth, csrnet_real.pth)"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


class CSRNetPretrained(nn.Module):
    """6-layer backend: 512->512->512->256->128->64  (csrnet_pretrained.pth)"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


# ── Model Loader ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    """Auto-detects architecture from checkpoint and loads weights."""
    print(f"  Reading checkpoint : {checkpoint_path.name}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if not isinstance(checkpoint, dict):
        return checkpoint.to(device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    has_b6   = "backend.6.weight" in state_dict
    b0_shape = state_dict.get("backend.0.weight", torch.zeros(1)).shape

    if has_b6 and b0_shape[0] == 512:
        print("  Architecture       : CSRNetPretrained (6-layer backend)")
        model = CSRNetPretrained().to(device)
    else:
        print("  Architecture       : CSRNet (3-layer backend)")
        model = CSRNet().to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Weights loaded     : strict=True ✓")
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded     : strict=False")
        if result.missing_keys:
            print(f"  Missing keys       : {result.missing_keys}")

    return model


# ── Image Helpers ─────────────────────────────────────────────────────────────

def resize_keep_aspect(img, max_size=1024):
    """Resize PIL image so longest side <= max_size, keeping aspect ratio."""
    w, h  = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def predict_patches(model, img, patch_size=512, stride=384):
    """
    Sliding-window patch inference.
    Runs the model on overlapping 512x512 tiles and stitches with averaging.
    Much more accurate than squeezing the full image to 224x224.
    """
    w, h    = img.size
    img_np  = np.array(img, dtype=np.float32) / 255.0
    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    img_chw = torch.tensor(
        ((img_np - mean) / std).transpose(2, 0, 1), dtype=torch.float32
    )

    density_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc  = np.zeros((h, w), dtype=np.float32)
    model.eval()

    # Small image — single forward pass
    if h <= patch_size and w <= patch_size:
        with torch.no_grad():
            out = model(img_chw.unsqueeze(0).to(device))
        out_np = torch.relu(out).squeeze().cpu().numpy()
        return cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # Sliding window
    y_starts = list(range(0, h - patch_size + 1, stride))
    x_starts = list(range(0, w - patch_size + 1, stride))
    if not y_starts or y_starts[-1] + patch_size < h:
        y_starts.append(max(0, h - patch_size))
    if not x_starts or x_starts[-1] + patch_size < w:
        x_starts.append(max(0, w - patch_size))

    for y0 in y_starts:
        for x0 in x_starts:
            y1, x1 = min(y0 + patch_size, h), min(x0 + patch_size, w)
            patch  = img_chw[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(patch)
            out_np = torch.relu(out).squeeze().cpu().numpy()
            out_up = cv2.resize(out_np, (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_LINEAR)
            density_acc[y0:y1, x0:x1] += out_up
            weight_acc [y0:y1, x0:x1] += 1.0

    return density_acc / np.maximum(weight_acc, 1e-6)


# ── Scale Factor ─────────────────────────────────────────────────────────────

def compute_scale_factor(model):
    """
    Computes per-image median ratio (actual / raw_sum) across test sets.
    Uses patch-based inference — consistent with video inference.
    """
    import scipy.io as sio
    from scipy.ndimage import gaussian_filter

    def load_gt(img_path):
        try:
            p        = Path(img_path)
            num      = p.stem.replace("IMG_", "")
            mat_path = p.parent.parent / "ground_truth" / f"GT_IMG_{num}.mat"
            if not mat_path.exists():
                return None
            mat    = sio.loadmat(str(mat_path))
            points = mat["image_info"][0][0][0][0][0]
            return len(points)
        except Exception:
            return None

    test_dirs = [
        DATA_ROOT.parent.parent / "dataset" / "archive" / "part_A_final" / "test_data",
        DATA_ROOT.parent.parent / "dataset" / "archive" / "part_B_final" / "test_data",
    ]

    ratios = []
    model.eval()
    print("  Scanning test images for scale calibration...")

    for test_dir in test_dirs:
        img_dir = test_dir / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            actual = load_gt(str(img_path))
            if not actual:
                continue
            try:
                img     = Image.open(img_path).convert("RGB")
                img     = resize_keep_aspect(img)
                density = predict_patches(model, img)
                raw_sum = float(density.sum())
                if raw_sum > 0:
                    ratios.append(actual / raw_sum)
            except Exception:
                continue

    if ratios:
        ratios   = np.array(ratios)
        m, s     = ratios.mean(), ratios.std()
        filtered = ratios[np.abs(ratios - m) < 2 * s]
        scale    = float(np.clip(np.median(filtered), 0.5, 10.0))
        print(f"  Samples used   : {len(filtered)} / {len(ratios)}")
        print(f"  Ratio range    : {filtered.min():.3f} – {filtered.max():.3f}")
        print(f"  Scale factor   : {scale:.4f}")
        return scale

    print("  WARNING: No test data found — using default scale 1.0")
    return 1.0


# ── HUD Drawing ──────────────────────────────────────────────────────────────

def draw_hud(frame, count, fps, frame_idx, total_frames, alert_threshold=300):
    """Draws count, FPS, progress bar and alert banner onto frame."""
    h, w = frame.shape[:2]

    # Top dark bar
    cv2.rectangle(frame, (0, 0), (w, 95), (15, 15, 15), -1)

    # Crowd count
    count_color = (0, 255, 0) if count < alert_threshold else (0, 80, 255)
    cv2.putText(frame,
                f"Crowd Count: {int(count)} people",
                (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, count_color, 2, cv2.LINE_AA)

    # FPS + progress
    progress = f"Frame {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame {frame_idx}"
    cv2.putText(frame,
                f"FPS: {fps:.1f}   {progress}",
                (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 160, 160), 1, cv2.LINE_AA)

    # Alert banner (top-right)
    if count >= alert_threshold:
        bw = 230
        cv2.rectangle(frame, (w - bw, 0), (w, 95), (0, 0, 180), -1)
        cv2.putText(frame, "HIGH DENSITY",
                    (w - bw + 12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ALERT",
                    (w - bw + 55, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 210, 255), 2, cv2.LINE_AA)

    # Bottom progress bar
    if total_frames > 0:
        bar_w = int(w * frame_idx / total_frames)
        cv2.rectangle(frame, (0, h - 6), (w, h),     (40, 40, 40),  -1)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), (0, 200, 100), -1)

    return frame


# ── Main Video Pipeline ───────────────────────────────────────────────────────

def video_predict(model, scale_factor, video_path,
                  save_output=False, frame_skip=2,
                  show_density=False, alert_threshold=300):
    """
    Full video crowd-counting pipeline.

    Args:
        model          : loaded CSRNet model
        scale_factor   : calibrated scale (from compute_scale_factor)
        video_path     : path to input .mp4 / .avi / .mov file
        save_output    : if True, saves annotated video to outputs/
        frame_skip     : run model every N frames (default 2 = 2x faster)
        show_density   : if True, shows density heatmap side-by-side
        alert_threshold: crowd count above which ALERT banner fires
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n  ERROR: Cannot open video → {video_path}")
        print("  Check the file path and make sure the file exists.\n")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  {'='*50}")
    print(f"  Video file     : {Path(video_path).name}")
    print(f"  Resolution     : {width} x {height}")
    print(f"  Input FPS      : {fps_in:.1f}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Processing     : every {frame_skip} frame(s)")
    print(f"  Density overlay: {'ON' if show_density else 'OFF'}")
    print(f"  Alert at count : {alert_threshold}+ people")
    print(f"  Save output    : {'YES' if save_output else 'NO'}")
    print(f"  {'='*50}")
    print(f"  Press  Q  to quit at any time\n")

    # ── Video Writer ─────────────────────────────────────────────────
    writer   = None
    out_path = None
    if save_output:
        out_dir  = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"video_output_{Path(video_path).stem}.mp4"
        out_w    = width * 2 if show_density else width
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(
            str(out_path), fourcc,
            fps_in / max(frame_skip, 1),
            (out_w, height)
        )
        print(f"  Saving to      : {out_path}\n")

    # ── State ────────────────────────────────────────────────────────
    frame_idx     = 0
    last_count    = 0.0
    last_density  = None
    count_history = []
    fps_display   = 0.0
    t_prev        = time.time()

    # ── Main Loop ────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run model every Nth frame only
        if frame_idx % frame_skip == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = resize_keep_aspect(pil_img, max_size=1024)

            raw_density  = predict_patches(model, pil_img)
            raw_count    = float(raw_density.sum())
            last_count   = float(np.clip(raw_count * scale_factor, 0, 5000))
            last_density = raw_density
            count_history.append(last_count)

            t_now       = time.time()
            fps_display = frame_skip / max(t_now - t_prev, 1e-6)
            t_prev      = t_now

        # Build display frame
        display = frame.copy()

        # Side-by-side density heatmap
        if show_density and last_density is not None:
            dn_norm    = last_density / (last_density.max() + 1e-6)
            dn_colored = cv2.applyColorMap(
                (dn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            dn_resized = cv2.resize(dn_colored, (width, height))

            # Label the density panel
            cv2.rectangle(dn_resized, (0, 0), (width, 36), (0, 0, 0), -1)
            cv2.putText(dn_resized, "Density Map",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 1, cv2.LINE_AA)

            display = np.hstack([display, dn_resized])

        # Draw HUD
        display = draw_hud(
            display, last_count, fps_display,
            frame_idx, total_frames, alert_threshold
        )

        # Save frame
        if writer is not None:
            writer.write(display)

        # Show
        cv2.imshow("Video Crowd Counting  [Q = quit]", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  Stopped by user.")
            break

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────
    if count_history:
        print(f"\n  {'='*50}")
        print(f"  SUMMARY")
        print(f"  {'='*50}")
        print(f"  Frames processed  : {len(count_history)}")
        print(f"  Avg crowd count   : {np.mean(count_history):.0f}")
        print(f"  Max crowd count   : {np.max(count_history):.0f}")
        print(f"  Min crowd count   : {np.min(count_history):.0f}")
        print(f"  Alerts triggered  : {sum(1 for c in count_history if c >= alert_threshold)}")
        if out_path:
            print(f"  Output saved to   : {out_path}")
        print(f"  {'='*50}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSRNet Video Crowd Counting Pipeline"
    )
    parser.add_argument("--video",         type=str,  required=True,
                        help="Path to input video file (.mp4 / .avi / .mov)")
    parser.add_argument("--model",         type=str,  default="pretrained",
                        help="pretrained | best | real  (default: pretrained)")
    parser.add_argument("--skip",          type=int,  default=2,
                        help="Run model every N frames (default: 2)")
    parser.add_argument("--show-density",  action="store_true",
                        help="Show density heatmap side-by-side")
    parser.add_argument("--save",          action="store_true",
                        help="Save annotated output video to outputs/")
    parser.add_argument("--alert",         type=int,  default=300,
                        help="Crowd count threshold for alert banner (default: 300)")
    parser.add_argument("--scale",         type=float, default=None,
                        help="Override scale factor manually (skip auto-calibration)")
    args = parser.parse_args()

    # ── STEP 1: Load Model ───────────────────────────────────────────
    print("[ STEP 1 ]  Loading Model...")
    model_candidates = {
        "pretrained": Path(MODEL_DIR) / "csrnet_pretrained.pth",
        "best":       Path(MODEL_DIR) / "csrnet_best.pth",
        "real":       Path(MODEL_DIR) / "csrnet_real.pth",
    }

    order      = [args.model] + [k for k in model_candidates if k != args.model]
    model_path = None
    for key in order:
        candidate = model_candidates.get(key)
        if candidate and candidate.exists():
            model_path = candidate
            print(f"  Using model    : {key}  ({candidate.name})")
            break

    if model_path is None:
        raise FileNotFoundError(
            "No model file found! Expected one of:\n" +
            "\n".join(f"  {v}" for v in model_candidates.values())
        )

    model = load_model(model_path)
    print(f"  Parameters     : {sum(p.numel() for p in model.parameters()):,}")

    # ── STEP 2: Scale Factor ─────────────────────────────────────────
    print("\n[ STEP 2 ]  Scale Factor...")
    if args.scale is not None:
        scale = args.scale
        print(f"  Using manual scale factor : {scale}")
    else:
        scale = compute_scale_factor(model)

    # ── STEP 3: Run Video Pipeline ───────────────────────────────────
    print("\n[ STEP 3 ]  Starting Video Pipeline...")
    video_predict(
        model          = model,
        scale_factor   = scale,
        video_path     = args.video,
        save_output    = args.save,
        frame_skip     = args.skip,
        show_density   = args.show_density,
        alert_threshold= args.alert,
    )

    print("=" * 52)
    print("  DONE!")
    print("=" * 52 + "\n")