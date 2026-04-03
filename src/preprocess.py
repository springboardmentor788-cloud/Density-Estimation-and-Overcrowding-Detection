from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import DATA_ROOT, PREPROCESSED_ROOT, PreprocessConfig


def preprocess_split(part: str = "A", split: str = "train") -> None:
    """
    Preprocess images for a given dataset part and split.

    Args:
        part: "A" or "B"
        split: "train" or "test"
    """
    img_dir = (
        DATA_ROOT / f"part_{part}_final" / f"{split}_data" / "images"
    )
    out_img_dir = PREPROCESSED_ROOT / f"part_{part}" / split / "images"

    if not img_dir.exists():
        print(f"[WARN] Input image directory does not exist: {img_dir}")
        print("       Ensure the ShanghaiTech dataset is placed correctly.")
        return

    out_img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in PreprocessConfig.IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print(f"[WARN] No images found in {img_dir}")
        return

    print(
        f"[INFO] Preprocessing {len(image_paths)} images from {img_dir} "
        f"-> {out_img_dir}"
    )

    for img_path in tqdm(image_paths, desc=f"part_{part} {split}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        # Resize to target size
        resized = cv2.resize(
            img,
            PreprocessConfig.TARGET_SIZE,
            interpolation=cv2.INTER_CUBIC,
        )

        # Normalize to [0, 1] (per-pixel) and save as 0–255 uint8 image.
        # The per-channel mean/std normalization will be applied in the Dataset.
        normalized = resized.astype(np.float32) / 255.0
        out_path = out_img_dir / img_path.name
        cv2.imwrite(str(out_path), (normalized * 255.0).astype(np.uint8))


def preprocess_all() -> None:
    """
    Run preprocessing for Part A and Part B, train and test splits.
    Safe to call even if some folders are missing; it will warn and skip.
    """
    for part in ("A", "B"):
        for split in ("train", "test"):
            preprocess_split(part=part, split=split)


if __name__ == "__main__":
    preprocess_all()

