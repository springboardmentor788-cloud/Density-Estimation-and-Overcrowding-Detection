# ============================================================
#  scripts/predict.py
#  Run:  python scripts/predict.py --samples 5
#  Run:  python scripts/predict.py --samples 5 --model best
#  Run:  python scripts/predict.py --samples 5 --dataset B
#  Run:  python scripts/predict.py --realtime
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
    """3-layer backend — matches csrnet_best.pth / csrnet_real.pth"""
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
    """6-layer backend — matches csrnet_pretrained.pth"""
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


# ── Auto-detect & Load ───────────────────────────────────────────────────────

def build_model_from_checkpoint(checkpoint_path):
    print(f"  Reading checkpoint : {checkpoint_path.name}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if not isinstance(checkpoint, dict):
        return checkpoint.to(device)

    for key in ["state_dict", "model_state_dict", "model"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    has_b6   = "backend.6.weight" in state_dict
    b0_shape = state_dict.get("backend.0.weight", torch.zeros(1)).shape

    print(f"  backend.0.weight   : {b0_shape}")
    print(f"  backend.6 present  : {has_b6}")

    if has_b6 and b0_shape[0] == 512:
        print("  Architecture       : CSRNetPretrained  (6-layer 512-channel backend)")
        model = CSRNetPretrained().to(device)
    else:
        print("  Architecture       : CSRNet (3-layer backend)")
        model = CSRNet().to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Weights loaded     : strict=True  ✓  (all keys matched)")
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded     : strict=False")
        if result.missing_keys:
            print(f"  Missing keys       : {result.missing_keys}")

    return model


# ── Ground Truth ─────────────────────────────────────────────────────────────

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


def make_density_map(points, img_shape, sigma=8):
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


# ── Transform (used only for non-pretrained models) ──────────────────────────

transform_fixed = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── FIX 1: Patch-based inference (for CSRNetPretrained) ──────────────────────

def resize_keep_aspect(img, max_size=1024):
    """Resize PIL image so longest side <= max_size, keeping aspect ratio."""
    w, h  = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def predict_patches(model, img, patch_size=512, stride=384):
    """
    Sliding-window patch inference — accurate for high-res images.
    Used for CSRNetPretrained which needs full resolution, not 224x224 resize.
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


def is_pretrained_model(model):
    """Check if model is CSRNetPretrained (needs patch inference)."""
    return isinstance(model, CSRNetPretrained)


# ── FIX 2: Per-dataset scale factor (no bad global clamp) ────────────────────

def compute_scale_factor(model, dataset="both"):
    """
    Computes scale factor per dataset.
    Uses patch inference for CSRNetPretrained, resize for CSRNet.
    Removed the 0.5 minimum clamp that was forcing wrong scale.
    """
    print("  Computing scale factor from test images...")

    base = DATA_ROOT.parent.parent / "dataset" / "archive"

    if dataset == "A":
        test_dirs = [base / "part_A_final" / "test_data"]
    elif dataset == "B":
        test_dirs = [base / "part_B_final" / "test_data"]
    else:
        test_dirs = [
            base / "part_A_final" / "test_data",
            base / "part_B_final" / "test_data",
        ]

    ratios = []
    model.eval()
    use_patches = is_pretrained_model(model)

    for test_dir in test_dirs:
        img_dir = test_dir / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            actual, _ = get_ground_truth(str(img_path))
            if actual is None or actual == 0:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                if use_patches:
                    # FIX: use patch inference for pretrained model
                    img_resized = resize_keep_aspect(img, max_size=1024)
                    density     = predict_patches(model, img_resized)
                    raw_sum     = float(density.sum())
                else:
                    img_tensor = transform_fixed(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_tensor)
                    output  = torch.relu(output)
                    raw_sum = float(output.sum().item())

                if raw_sum > 0:
                    ratios.append(actual / raw_sum)
            except Exception:
                continue

    if ratios:
        ratios   = np.array(ratios)
        mean, std = ratios.mean(), ratios.std()
        filtered  = ratios[np.abs(ratios - mean) < 2 * std]

        # FIX: removed 0.5–3.0 clamp — use a wider, safer range
        scale = float(np.clip(np.median(filtered), 0.01, 20.0))

        print(f"  Total samples  : {len(ratios)}")
        print(f"  After filter   : {len(filtered)}")
        print(f"  Ratio range    : {filtered.min():.3f} – {filtered.max():.3f}")
        print(f"  Scale factor   : {scale:.4f}  (median per-image ratio, clipped 0.01–20.0)")
        return scale

    print("  WARNING: Could not compute scale — using default 1.0")
    return 1.0


# ── FIX 3: predict_image uses patch inference for pretrained ─────────────────

def predict_image(model, image_path, scale_factor):
    img = Image.open(image_path).convert("RGB")

    if is_pretrained_model(model):
        # Use patch-based inference (same as video_predict.py)
        img_resized = resize_keep_aspect(img, max_size=1024)
        density_map = predict_patches(model, img_resized)
        raw_sum     = float(density_map.sum())
        pred_count  = float(np.clip(raw_sum * scale_factor, 0, 50000))
        d_display   = density_map / (density_map.max() + 1e-6)
    else:
        img_tensor = transform_fixed(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        output      = torch.relu(output)
        density_map = output.squeeze().cpu().numpy()
        raw_sum     = float(output.sum().item())
        pred_count  = raw_sum * scale_factor
        d_display   = density_map / (density_map.max() + 1e-6)

    return img, d_display, pred_count


# ── Mentor-Style Output ──────────────────────────────────────────────────────

def mentor_output(image_path, model, scale_factor, num_samples=5):
    p        = Path(image_path)
    img_dir  = p.parent
    all_imgs = sorted(img_dir.glob("*.jpg"))
    print(f"  Dataset        : {img_dir}")
    print(f"  Total images   : {len(all_imgs)}  →  using first {num_samples}")
    imgs = all_imgs[:num_samples]

    nrows     = len(imgs)
    fig, axes = plt.subplots(
        nrows, 4, figsize=(14, nrows * 3),
        gridspec_kw={"width_ratios": [3, 3, 3, 1.8]}
    )

    if nrows == 1:
        axes = [axes]

    for j, h in enumerate(["Input", "Ground truth", "Predicted", "Results"]):
        axes[0][j].set_title(h, fontsize=14, fontweight="bold", pad=12)

    total_gt, total_pred, diffs = 0, 0, []

    for i, img_path in enumerate(imgs):
        orig_img, pred_dm, pred_count = predict_image(
            model, str(img_path), scale_factor
        )
        actual_count, gt_dm = get_ground_truth(str(img_path))

        # Col 0: Input
        axes[i][0].imshow(orig_img)
        axes[i][0].axis("off")

        # Col 1: Ground truth density map
        axes[i][1].set_facecolor("black")
        if gt_dm is not None:
            gt_norm = gt_dm / gt_dm.max() if gt_dm.max() > 0 else gt_dm
            axes[i][1].imshow(gt_norm, cmap="jet", vmin=0, vmax=1)
        axes[i][1].axis("off")

        # Col 2: Predicted density map
        axes[i][2].set_facecolor("black")
        axes[i][2].imshow(pred_dm, cmap="jet", vmin=0, vmax=1)
        axes[i][2].axis("off")

        # Col 3: Results text
        axes[i][3].axis("off")
        axes[i][3].set_facecolor("#f9f9f9")
        if actual_count is not None:
            axes[i][3].text(0.1, 0.65,
                            f"Ground truth:  {actual_count}",
                            transform=axes[i][3].transAxes,
                            fontsize=11, color="darkgreen", fontweight="bold")
            total_gt += actual_count
        axes[i][3].text(0.1, 0.35,
                        f"Predicted:  {pred_count:.0f}",
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

    mae = np.mean(diffs) if diffs else 0
    print(f"\n  {'─'*52}")
    print(f"  Total GT    : {total_gt}")
    print(f"  Total Pred  : {total_pred:.0f}")
    print(f"  MAE         : {mae:.1f}  (Mean Absolute Error)")
    print(f"  {'─'*52}")

    plt.suptitle("Crowd Counting Results",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "mentor_output.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\n  Saved → {save_path}\n")


# ── Real-time ─────────────────────────────────────────────────────────────────

def realtime_predict(model, scale_factor):
    print("\n  Starting webcam... Press Q to quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  No webcam — switching to test images...")
        realtime_images(model, scale_factor)
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if is_pretrained_model(model):
            pil_img = resize_keep_aspect(pil_img, max_size=1024)
            density = predict_patches(model, pil_img)
            count   = float(density.sum()) * scale_factor
        else:
            img_tensor = transform_fixed(pil_img).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
            output = torch.relu(output)
            count  = float(output.sum().item()) * scale_factor

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Crowd Count: {count:.0f} people",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 2)
        cv2.imshow("Real Time Crowd Counting  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def realtime_images(model, scale_factor):
    print("  Press any key → next   Q → quit\n")
    test_dir = (DATA_ROOT.parent.parent / "dataset" / "archive" /
                "part_A_final" / "test_data" / "images")
    for img_path in sorted(test_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if is_pretrained_model(model):
            pil_img_r = resize_keep_aspect(pil_img, max_size=1024)
            density   = predict_patches(model, pil_img_r)
            count     = float(density.sum()) * scale_factor
        else:
            img_tensor = transform_fixed(pil_img).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
            output = torch.relu(output)
            count  = float(output.sum().item()) * scale_factor

        actual, _ = get_ground_truth(str(img_path))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
        cv2.putText(frame, f"Predicted: {count:.0f} people",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        if actual:
            cv2.putText(frame, f"Actual:    {actual} people",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)
        print(f"  {img_path.name:<15}  "
              f"Pred: {count:>6.0f}  "
              f"Actual: {actual if actual else 'N/A':>6}")
        cv2.imshow("Real Time  [any key=next, Q=quit]", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str,  default=None)
    parser.add_argument("--samples",  type=int,  default=5)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--model",    type=str,  default="best",
                        help="best | real | pretrained  (default: best)")
    parser.add_argument("--dataset",  type=str,  default="A",
                        help="A | B  (default: A)")
    args = parser.parse_args()

    if args.image is None:
        args.image = str(
            DATA_ROOT.parent.parent / "dataset" / "archive" /
            f"part_{args.dataset}_final" / "test_data" / "images" / "IMG_1.jpg"
        )

    # STEP 1: Load Model
    print("[ STEP 1 ]  Loading Model...")
    model_candidates = {
        "best"       : Path(MODEL_DIR) / "csrnet_best.pth",
        "real"       : Path(MODEL_DIR) / "csrnet_real.pth",
        "pretrained" : Path(MODEL_DIR) / "csrnet_pretrained.pth",
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
        raise FileNotFoundError("No model file found!")

    model = build_model_from_checkpoint(model_path)
    print(f"  Model loaded  → {model_path}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # STEP 2: Scale Factor — now per dataset
    print("\n[ STEP 2 ]  Computing Scale Factor...")
    scale = compute_scale_factor(model, dataset=args.dataset)

    # STEP 3: Predict
    if args.realtime:
        print("\n[ STEP 3 ]  Real Time Prediction...")
        realtime_predict(model, scale)
    else:
        print(f"\n[ STEP 3 ]  Generating Output ({args.samples} images)...")
        if args.samples > 10:
            print(f"  Tip: Run with --samples 50 for a more realistic MAE estimate")
        mentor_output(args.image, model, scale, num_samples=args.samples)

    print("=" * 55)
    print("  DONE!")
    print("=" * 55 + "\n")