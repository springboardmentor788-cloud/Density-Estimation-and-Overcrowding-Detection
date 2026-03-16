import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_ROOT, OUTPUT_DIR

OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_all_paths():
    folders = [
        DATA_ROOT / "part_A_final" / "train_data" / "images",
        DATA_ROOT / "part_A_final" / "test_data"  / "images",
        DATA_ROOT / "part_B_final" / "train_data" / "images",
        DATA_ROOT / "part_B_final" / "test_data"  / "images",
    ]
    all_paths = []
    for folder in folders:
        if Path(folder).exists():
            all_paths += [str(p) for p in Path(folder).iterdir()
                          if p.suffix.lower() in IMG_EXTS]
    print(f"\nTotal images in dataset : {len(all_paths)}\n")
    return all_paths

def plot_density_map(image_paths):
    from scipy.stats import gaussian_kde
    widths, heights = [], []
    for p in image_paths:
        w, h = Image.open(p).size
        widths.append(w); heights.append(h)

    xy  = np.vstack([widths, heights])
    kde = gaussian_kde(xy)
    xi, yi = np.mgrid[
        min(widths)-20:max(widths)+20:150j,
        min(heights)-20:max(heights)+20:150j
    ]
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(xi, yi, zi, levels=20, cmap="plasma")
    ax.scatter(widths, heights, s=10, c="white", alpha=0.4, zorder=5)
    ax.axvline(224, color="cyan", lw=2, linestyle="--", label="resize W=224")
    ax.axhline(224, color="lime", lw=2, linestyle="--", label="resize H=224")
    plt.colorbar(cf, ax=ax, label="Density")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Density Map - Original Image Sizes")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "density_map.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved -> {out}")

def visualize_samples(image_paths, n=8):
    sample = random.sample(image_paths, min(n, len(image_paths)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.2))
    axes = axes.flatten()

    for i, path in enumerate(sample):
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        ax   = axes[i]
        ax.imshow(img)
        ax.set_title(f"Orig: {w}x{h}\n-> 224x224", fontsize=7)
        rect = mpatches.Rectangle((0,0), w-1, h-1,
                                   linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(4, 18, f"pos:(0,0)  size:{w}x{h}",
                color="yellow", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.5, pad=2))
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"Sample Images  (Total = {len(image_paths)})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "sample_images.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved -> {out}")

if __name__ == "__main__":
    all_paths = collect_all_paths()
    if not all_paths:
        print("No images found!")
        sys.exit(1)
    plot_density_map(all_paths)
    visualize_samples(all_paths, n=8)
    print("\nDone! Check outputs/ folder.")