import sys
from pathlib import Path
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_ROOT, PREPROCESSED_ROOT, IMAGE_SIZE, PARTS, SPLITS

def preprocess_split(part, split):
    img_dir = DATA_ROOT / f"part_{part}_final" / f"{split}_data" / "images"
    out_dir = PREPROCESSED_ROOT / f"part_{part}" / split / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []
    if img_dir.exists():
        image_files = [f for f in sorted(img_dir.iterdir())
                       if f.suffix.lower() in IMG_EXTS]

    if not image_files:
        print(f"  [SKIP] No images in: {img_dir}")
        return 0

    print(f"\n  Part {part} - {split}  ({len(image_files)} images)")
    processed = 0
    for img_path in tqdm(image_files, desc=f"Part {part}/{split}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        cv2.imwrite(str(out_dir / img_path.name), img_resized)
        processed += 1

    print(f"  Saved {processed} images to {out_dir}")
    return processed

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  PREPROCESSING DATASET")
    print(f"  Target size : {IMAGE_SIZE[1]} x {IMAGE_SIZE[0]}")
    print("="*50)

    total = 0
    for part in PARTS:
        for split in SPLITS:
            total += preprocess_split(part, split)

    print("\n" + "="*50)
    print(f"  Total preprocessed : {total}")
    print("="*50)