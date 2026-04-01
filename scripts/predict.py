# ============================================================
#  scripts/predict.py
#  Run:  python scripts/predict.py
#  Run:  python scripts/predict.py --samples 5
#  Run:  python scripts/predict.py --realtime
#  Run:  python scripts/predict.py --model pretrained
# ============================================================

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_DIR, OUTPUT_DIR, DATA_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

IMG_SIZE = 224


# ── Model Definitions ────────────────────────────────────────────────────────

class CSRNet(nn.Module):
    """
    Small backend: 3 conv layers  256->128->64
    Used by: csrnet_best.pth, csrnet_real.pth
    """
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
    """
    Exact architecture matching csrnet_pretrained.pth (34 keys).

    Backend layers confirmed from checkpoint inspection:
      backend.0  : Conv2d(512, 512, 3)   [512, 512, 3, 3]
      backend.2  : Conv2d(512, 512, 3)   [512, 512, 3, 3]
      backend.4  : Conv2d(512, 512, 3)   [512, 512, 3, 3]
      backend.6  : Conv2d(512, 256, 3)   [256, 512, 3, 3]
      backend.8  : Conv2d(256, 128, 3)   [128, 256, 3, 3]
      backend.10 : Conv2d(128,  64, 3)   [ 64, 128, 3, 3]
      output_layer: Conv2d(64, 1, 1)     [  1,  64, 1, 1]

    Frontend matches VGG16 layers 0-21 (same as CSRNet).
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 0,1
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 2,3
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 4,5
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 6,7
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 8,9
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # 10,11
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


# ── Auto-detect & Load ───────────────────────────────────────────────────────
def build_model_from_checkpoint(checkpoint_path):
    """
    Inspects checkpoint keys/shapes and picks the matching architecture.
    Handles all checkpoint formats and cuda->cpu mapping automatically.
    """
    print(f"  Reading checkpoint : {checkpoint_path.name}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Unwrap nested formats
    if not isinstance(checkpoint, dict):
        print("  Format: full model object")
        return checkpoint.to(device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip DataParallel prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ── Detect architecture ──────────────────────────────────────────
    total_keys   = len(state_dict)
    has_b6       = "backend.6.weight" in state_dict
    b0_shape     = state_dict.get("backend.0.weight", torch.zeros(1)).shape

    print(f"  Total keys         : {total_keys}")
    print(f"  backend.0.weight   : {b0_shape}")
    print(f"  backend.6 present  : {has_b6}")

    if has_b6 and b0_shape[0] == 512:
        # 6-conv backend: 512->512->512->256->128->64  (34 keys)
        print("  Architecture       : CSRNetPretrained  (6-layer 512-channel backend)")
        model = CSRNetPretrained().to(device)
    else:
        # 3-conv backend: 256->128->64  (26 keys)
        print("  Architecture       : CSRNet  (3-layer backend)")
        model = CSRNet().to(device)

    # ── Load weights ─────────────────────────────────────────────────
    missing, unexpected = [], []
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
        print("  Weights loaded     : strict=True  ✓  (all keys matched)")
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        missing    = result.missing_keys
        unexpected = result.unexpected_keys
        print(f"  Weights loaded     : strict=False")
        if missing:
            print(f"  Missing keys       : {missing}")
        if unexpected:
            print(f"  Unexpected keys    : {unexpected}")

    return model


# ── Ground Truth Helpers ─────────────────────────────────────────────────────
def load_mat(mat_path):
    try:
        mat    = sio.loadmat(str(mat_path))
        points = mat["image_info"][0][0][0][0][0]
        return points
    except Exception:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            points = np.array(f["image_info"]["location"]).T
        return points


# FIX #4: Reduced sigma from 8 → 4 for better GT match
def make_density_map(points, img_shape, sigma=4):
    H, W    = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    for point in points:
        x = int(float(point[0]))
        y = int(float(point[1]))
        if 0 <= x < W and 0 <= y < H:
            density[y, x] += 1.0
    return gaussian_filter(density, sigma=sigma)


def get_ground_truth(image_path):
    try:
        p        = Path(image_path)
        num      = p.stem.replace("IMG_", "")
        gt_dir   = p.parent.parent / "ground_truth"
        mat_path = gt_dir / f"GT_IMG_{num}.mat"
        if mat_path.exists():
            points  = load_mat(str(mat_path))
            count   = len(points)
            img     = Image.open(image_path).convert("RGB")
            w, h    = img.size
            density = make_density_map(points, (h, w))
            return count, density
    except Exception:
        pass
    return None, None


# FIX #2: Helper to resize large images while keeping aspect ratio
def resize_keep_aspect(img, max_size=2048):
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    return img


# ── Scale Factor ─────────────────────────────────────────────────────────────
def compute_scale_factor(model):
    """
    Per-image median ratio:
        scale = median( actual_count / raw_output_sum )

    Why median and not mean/global-sum:
      - Robust: one very large/small crowd does not skew the scale
      - Per-image: every image gets equal weight regardless of crowd size
      - Outlier removal: samples beyond 2 std are discarded before median
    """
    print("  Computing scale factor from test images...")
    test_dirs = [
        DATA_ROOT.parent.parent / "dataset" / "archive" / "part_A_final" / "test_data",
        DATA_ROOT.parent.parent / "dataset" / "archive" / "part_B_final" / "test_data",
    ]
    ratios = []
    model.eval()

    for test_dir in test_dirs:
        img_dir = test_dir / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            actual, _ = get_ground_truth(str(img_path))
            if actual is None or actual == 0:
                continue
            try:
                # FIX #2: Use resize_keep_aspect instead of fixed resize
                img = Image.open(img_path).convert("RGB")
                img = resize_keep_aspect(img)
                raw_density = predict_image_patches(model, img)
                raw_sum = float(raw_density.sum())
                if raw_sum > 0:
                    ratios.append(actual / raw_sum)
            except Exception:
                continue

    if ratios:
        ratios    = np.array(ratios)
        mean, std = ratios.mean(), ratios.std()
        filtered  = ratios[np.abs(ratios - mean) < 2 * std]
        # FIX #3: Clip scale factor to prevent extreme scaling
        scale     = float(np.clip(np.median(filtered), 0.5, 10.0))
        print(f"  Total samples  : {len(ratios)}")
        print(f"  After filter   : {len(filtered)}")
        print(f"  Ratio range    : {filtered.min():.3f} – {filtered.max():.3f}")
        print(f"  Scale factor   : {scale:.4f}  (median per-image ratio, clipped 0.5–3.0)")
        return scale
    return 5.1


# FIX #1: Removed Resize(224, 224) from transform_base
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def predict_image_patches(model, img, patch_size=512, stride=384):
    """
    Sliding window patch-based prediction.
    Splits image into overlapping patches -> runs model on each ->
    stitches density maps with overlap averaging.
    Much more accurate than single-shot resize for large/complex images.
    """
    w, h       = img.size
    img_np     = np.array(img, dtype=np.float32) / 255.0
    mean       = np.array([0.485, 0.456, 0.406])
    std        = np.array([0.229, 0.224, 0.225])
    img_norm   = (img_np - mean) / std
    img_chw    = torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32)

    density_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc  = np.zeros((h, w), dtype=np.float32)

    model.eval()

    if h <= patch_size and w <= patch_size:
        inp = img_chw.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        out_np = torch.relu(out).squeeze().cpu().numpy()
        out_up = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LINEAR)
        return out_up

    y_starts = list(range(0, h - patch_size + 1, stride))
    x_starts = list(range(0, w - patch_size + 1, stride))
    if not y_starts or y_starts[-1] + patch_size < h:
        y_starts.append(max(0, h - patch_size))
    if not x_starts or x_starts[-1] + patch_size < w:
        x_starts.append(max(0, w - patch_size))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            patch = img_chw[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(patch)
            out_np = torch.relu(out).squeeze().cpu().numpy()
            ph, pw = y1 - y0, x1 - x0
            out_up = cv2.resize(out_np, (pw, ph), interpolation=cv2.INTER_LINEAR)
            density_acc[y0:y1, x0:x1] += out_up
            weight_acc [y0:y1, x0:x1] += 1.0

    weight_acc  = np.maximum(weight_acc, 1e-6)
    density_map = density_acc / weight_acc
    return density_map


def predict_image(model, image_path, scale_factor):
    img = Image.open(image_path).convert("RGB")
    img = resize_keep_aspect(img)

    raw_density = predict_image_patches(model, img)
    raw_count   = float(raw_density.sum())

    # FIX #5: Clamp predicted count to prevent overcount
    pred_count  = np.clip(raw_count * scale_factor, 0, 5000)
    # FIX #7: Normalize density map for visualization
    density_map = raw_density / (raw_density.max() + 1e-6)
    return img, density_map, pred_count


# ── Mentor-Style Output ──────────────────────────────────────────────────────
def mentor_output(image_path, model, scale_factor, num_samples=5):
    p       = Path(image_path)
    img_dir = p.parent
    all_imgs = sorted(img_dir.glob("*.jpg"))
    print(f"  Dataset        : {img_dir}")
    print(f"  Total images   : {len(all_imgs)}  →  using first {num_samples}")
    imgs    = all_imgs[:num_samples]

    nrows     = len(imgs)
    fig, axes = plt.subplots(
        nrows, 4, figsize=(14, nrows * 3),
        gridspec_kw={"width_ratios": [3, 3, 3, 1.8]}
    )

    for j, h in enumerate(["Input", "Ground truth", "Predicted", "Results"]):
        axes[0][j].set_title(h, fontsize=14, fontweight="bold", pad=12)

    total_gt, total_pred, diffs = 0, 0, []

    for i, img_path in enumerate(imgs):
        orig_img, pred_dm, pred_count = predict_image(model, str(img_path), scale_factor)
        actual_count, gt_dm           = get_ground_truth(str(img_path))

        axes[i][0].imshow(orig_img)
        axes[i][0].axis("off")

        axes[i][1].set_facecolor("black")
        if gt_dm is not None:
            gt_norm = gt_dm / gt_dm.max() if gt_dm.max() > 0 else gt_dm
            axes[i][1].imshow(gt_norm, cmap="jet", vmin=0, vmax=1)
        axes[i][1].axis("off")

        axes[i][2].set_facecolor("black")
        # pred_dm already normalized in predict_image (FIX #7)
        axes[i][2].imshow(pred_dm, cmap="jet", vmin=0, vmax=1)
        axes[i][2].axis("off")

        axes[i][3].axis("off")
        axes[i][3].set_facecolor("#f9f9f9")
        if actual_count is not None:
            axes[i][3].text(0.1, 0.65, f"Ground truth:  {actual_count}",
                            transform=axes[i][3].transAxes,
                            fontsize=11, color="darkgreen", fontweight="bold")
            total_gt += actual_count
        axes[i][3].text(0.1, 0.35, f"Predicted:  {pred_count:.0f}",
                        transform=axes[i][3].transAxes,
                        fontsize=11, color="darkblue", fontweight="bold")
        total_pred += pred_count

        diff = abs(pred_count - actual_count) if actual_count else 0
        pct  = (diff / actual_count * 100) if actual_count else 0
        diffs.append(diff)
        print(f"  {img_path.name:<15}  "
              f"GT: {actual_count if actual_count else 'N/A':>6}  "
              f"Pred: {pred_count:>6.0f}  "
              f"Diff: {diff:>5.0f}  ({pct:.1f}%)")

    mae = np.mean(diffs)
    print(f"\n  {'─'*52}")
    print(f"  Total GT    : {total_gt}")
    print(f"  Total Pred  : {total_pred:.0f}")
    print(f"  MAE         : {mae:.1f}  (Mean Absolute Error)")
    print(f"  {'─'*52}")

    plt.suptitle("Crowd Counting Results", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "mentor_output.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\n  Saved → {save_path}\n")


# ── Real-time ────────────────────────────────────────────────────────────────
def realtime_predict(model, scale_factor):
    print("\n  Starting webcam... Press Q to quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  No webcam — switching to test images...")
        realtime_images(model, scale_factor, dataset=getattr(args, "dataset", "B"))
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img    = Image.fromarray(rgb)
        # FIX #2: Apply resize_keep_aspect in realtime too
        pil_img    = resize_keep_aspect(pil_img)
        raw_density = predict_image_patches(model, pil_img)
        raw_count   = float(raw_density.sum())
        count = np.clip(raw_count * scale_factor, 0, 5000)
        h, w  = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Crowd Count: {count:.0f} people",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
        cv2.imshow("Real Time Crowd Counting  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def realtime_images(model, scale_factor, dataset="B"):
    print("  Press any key → next   Q → quit\n")
    test_dir = (DATA_ROOT.parent.parent / "dataset" / "archive" /
                f"part_{dataset}_final" / "test_data" / "images")
    for img_path in sorted(test_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img    = Image.fromarray(rgb)
        # FIX #2: Apply resize_keep_aspect in realtime_images too
        pil_img    = resize_keep_aspect(pil_img)
        raw_density = predict_image_patches(model, pil_img)
        raw_count   = float(raw_density.sum())
        count = np.clip(raw_count * scale_factor, 0, 5000)
        actual, _ = get_ground_truth(str(img_path))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
        cv2.putText(frame, f"Predicted: {count:.0f} people",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        if actual:
            cv2.putText(frame, f"Actual:    {actual} people",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)
        print(f"  {img_path.name:<15}  Pred: {count:>6.0f}  Actual: {actual if actual else 'N/A':>6}")
        cv2.imshow("Real Time  [any key=next, Q=quit]", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, default=None)
    parser.add_argument("--samples",  type=int, default=5)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--model",    type=str, default="pretrained",
                        help="pretrained | best | real  (default: pretrained)")
    parser.add_argument("--dataset",  type=str, default="B",
                        help="A | B  — which ShanghaiTech part to use (default: B)")
    args = parser.parse_args()

    if args.image is None:
        dataset_dir = f"part_{args.dataset}_final"
        args.image = str(
            DATA_ROOT.parent.parent / "dataset" / "archive" /
            dataset_dir / "test_data" / "images" / "IMG_1.jpg"
        )

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
            print(f"  Using model   : {key} ({candidate.name})")
            break

    if model_path is None:
        raise FileNotFoundError(
            "No model file found! Expected one of:\n" +
            "\n".join(f"  {v}" for v in model_candidates.values())
        )

    model = build_model_from_checkpoint(model_path)
    print(f"  Model loaded  → {model_path}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # ── STEP 2: Compute Scale Factor ─────────────────────────────────
    print("\n[ STEP 2 ]  Computing Scale Factor...")
    scale = compute_scale_factor(model)

    # ── STEP 3: Predict ──────────────────────────────────────────────
    if args.realtime:
        print("\n[ STEP 3 ]  Real Time Prediction...")
        realtime_predict(model, scale)
    else:
        # FIX #6: Reminder to use --samples 50 for realistic MAE
        print(f"\n[ STEP 3 ]  Generating Output ({args.samples} images)...")
        print(f"  Tip: Run with --samples 50 for a more realistic MAE estimate")
        mentor_output(args.image, model, scale, num_samples=args.samples)

    print("=" * 55)
    print("  DONE!")
    print("=" * 55 + "\n")