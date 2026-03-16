import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import PART_A_TRAIN, OUTPUT_DIR

def load_image(path):
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    img = Image.open(img_path).convert("RGB")
    print(f"Image loaded successfully")
    return img

def visualize_image(img, title="Crowd Image"):
    w, h = img.size
    arr  = np.array(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(arr)
    axes[0].set_title(f"Original  [{w} x {h}]")
    axes[0].axis("off")
    rect = patches.Rectangle((0,0), w-1, h-1,
                               linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].text(6, 20,
        f"pos: (0, 0)\nsize: {w} x {h}\nshape: {arr.shape}",
        color="yellow", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.6, pad=3))

    img_resized = img.resize((224, 224))
    axes[1].imshow(img_resized)
    axes[1].set_title("After Resize  [224 x 224]")
    axes[1].axis("off")
    axes[1].text(4, 18,
        f"size: 224 x 224\nshape: {np.array(img_resized).shape}",
        color="yellow", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.6, pad=3))

    plt.tight_layout()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "visualized_image.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    search_folder = Path(PART_A_TRAIN)
    image_path = None

    if search_folder.exists():
        for f in sorted(search_folder.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                image_path = str(f)
                break

    if image_path is None:
        print("No images found!")
    else:
        image = load_image(image_path)
        visualize_image(image, title="Crowd Image")