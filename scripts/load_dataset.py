import sys
import os
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import (PART_A_TRAIN, PART_A_TEST,
                    PART_B_TRAIN, PART_B_TEST)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def load_images_from_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        print(f"  [WARNING] Folder not found: {folder}")
        return []
    paths = [str(f) for f in folder.iterdir()
             if f.suffix.lower() in IMG_EXTS]
    return sorted(paths)

if __name__ == "__main__":
    splits = {
        "Part A - Train" : PART_A_TRAIN,
        "Part A - Test"  : PART_A_TEST,
        "Part B - Train" : PART_B_TRAIN,
        "Part B - Test"  : PART_B_TEST,
    }

    all_paths = []
    print("\n" + "="*50)
    print("  LOADING DATASET")
    print("="*50)

    for name, folder in splits.items():
        paths = load_images_from_folder(Path(folder))
        print(f"  {name:<20} : {len(paths):>4} images")
        all_paths.extend(paths)

    print("-"*50)
    print(f"  Total images found : {len(all_paths)}")
    print("="*50)

    if all_paths:
        img = Image.open(all_paths[0]).convert("RGB")
        w, h = img.size
        print(f"\n  Sample : {all_paths[0]}")
        print(f"  Original size  : {w} x {h}  (W x H)")
        print(f"  After resize   : 224 x 224")
    else:
        print("\n  No images found - add images to data folder first!")

    print("\n[load_dataset.py]  Done.\n")