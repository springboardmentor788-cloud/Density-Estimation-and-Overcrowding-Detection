import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from config import DATA_ROOT
from dataset import (
    CrowdDataset,
    get_transforms,
    get_dataloader,
    get_optimizer,
    predict_size,
)

if __name__ == "__main__":

    IMAGE_DIR = DATA_ROOT / "part_A_final" / "train_data" / "images"

    print("\n" + "="*50)
    print("  PYTORCH DATALOADER PIPELINE  –  main.py")
    print("="*50)

    print("\n[ STEP 1 ]  Predict Image Size")
    print("-"*50)
    first_image = list(IMAGE_DIR.iterdir())[0]
    predict_size(str(first_image))

    print("[ STEP 2 ]  Create Custom PyTorch Dataset")
    print("-"*50)
    dataset = CrowdDataset(IMAGE_DIR, transform=get_transforms("train"))
    print(f"  Total images  : {len(dataset)}")
    img_tensor, _ = dataset[0]
    print(f"  Tensor shape  : {img_tensor.shape}  (C, H, W)")
    print(f"  Original size : {dataset.get_original_size(0)}")
    info = dataset.get_image_info(0)
    print(f"  After resize  : {info['after_resize']}")
    print(f"  Image mode    : {info['mode']}")

    print("\n[ STEP 3 ]  DataLoader  –  Batch Size 8")
    print("-"*50)
    loader8 = get_dataloader(IMAGE_DIR, mode="train", batch_size=8)
    batch, _ = next(iter(loader8))
    print(f"  Batch shape   : {batch.shape}")
    print(f"  Min pixel     : {batch.min():.3f}")
    print(f"  Max pixel     : {batch.max():.3f}")

    print("\n[ STEP 4 ]  DataLoader  –  Batch Size 16")
    print("-"*50)
    loader16 = get_dataloader(IMAGE_DIR, mode="train", batch_size=16)
    batch16, _ = next(iter(loader16))
    print(f"  Batch shape   : {batch16.shape}")

    print("\n[ STEP 5 ]  Pipeline Summary")
    print("-"*50)
    print("""
  Raw Image (1024 x 768)
       ↓  Resize      →  224 x 224
       ↓  RandomFlip  →  50% chance
       ↓  ColorJitter →  brightness/contrast
       ↓  ToTensor    →  0-255 to 0.0-1.0
       ↓  Normalize   →  center around 0
       ↓
  Tensor (3, 224, 224)
       ↓
  Batch (8, 3, 224, 224)
       ↓
  Ready for Model!
    """)

    print("[ STEP 6 ]  3 Gradient Descent Methods")
    print("-"*50)
    model = torch.nn.Linear(10, 2)
    opt1 = get_optimizer(model, "sgd")
    opt2 = get_optimizer(model, "adam")
    opt3 = get_optimizer(model, "adamw")
    print(f"  1. {opt1.__class__.__name__}   → simple")
    print(f"  2. {opt2.__class__.__name__}  → recommended ✅")
    print(f"  3. {opt3.__class__.__name__} → large models")

    print("\n" + "="*50)
    print("  ALL STEPS COMPLETE!")
    print("="*50 + "\n")