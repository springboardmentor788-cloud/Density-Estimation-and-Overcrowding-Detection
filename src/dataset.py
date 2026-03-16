import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from config import (DATA_ROOT, IMAGE_SIZE,
                    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)



def get_transforms(mode="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])



def predict_size(image_path):
    img  = Image.open(image_path).convert("RGB")
    w, h = img.size
    print("\n" + "="*45)
    print("  IMAGE SIZE PREDICTION")
    print("="*45)
    print(f"  Original size      : {w} x {h}  (W x H)")
    print(f"  After Resize       : 224 x 224  (W x H)")
    print(f"  Tensor shape       : (3, 224, 224)  (C, H, W)")
    print(f"  In a batch of 8    : (8, 3, 224, 224)")
    print(f"  Pixel range before : 0 - 255")
    print(f"  Pixel range after  : 0.0 - 1.0")
    print("="*45 + "\n")
    return {
        "original"     : (w, h),
        "after_resize" : (224, 224),
        "tensor_shape" : (3, 224, 224),
        "batch_shape"  : (8, 3, 224, 224),
    }



class CrowdDataset(Dataset):
   

    def __init__(self, image_dir, transform=None):
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_dir   = Path(image_dir)
        self.transform   = transform
        self.image_paths = []
        if self.image_dir.exists():
            self.image_paths = sorted([
                f for f in self.image_dir.iterdir()
                if f.suffix.lower() in IMG_EXTS
            ])
        print(f"[CrowdDataset]  Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path  = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path)

    def get_original_size(self, index):
        img = Image.open(self.image_paths[index])
        return img.size

    def get_image_info(self, index):
        path = self.image_paths[index]
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        return {
            "path"          : str(path),
            "original_size" : (w, h),
            "after_resize"  : (224, 224),
            "mode"          : img.mode,
        }



def get_dataloader(image_dir, mode="train", batch_size=None):
    dataset = CrowdDataset(image_dir, transform=get_transforms(mode))
    bs      = batch_size if batch_size else BATCH_SIZE
    loader  = DataLoader(
        dataset,
        batch_size  = bs,
        shuffle     = (mode == "train"),
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        drop_last   = False,
    )
    print(f"[DataLoader]  Batch size    : {bs}")
    print(f"[DataLoader]  Total batches : {len(loader)}")
    return loader




def get_optimizer(model, method="adam", lr=0.001):
    if method == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif method == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown: {method}. Use sgd/adam/adamw")




if __name__ == "__main__":
    IMAGE_DIR = DATA_ROOT / "part_A_final" / "train_data" / "images"

    print("\n" + "="*50)
    print("  PYTORCH PIPELINE  –  dataset.py")
    print("="*50)

    print("\n[ STEP 1 ]  Predict Image Size")
    first_image = list(IMAGE_DIR.iterdir())[0]
    predict_size(str(first_image))

    print("[ STEP 2 ]  Create Custom Dataset")
    dataset = CrowdDataset(IMAGE_DIR, transform=get_transforms("train"))
    print(f"  Total images  : {len(dataset)}")
    img_tensor, _ = dataset[0]
    print(f"  Tensor shape  : {img_tensor.shape}")
    info = dataset.get_image_info(0)
    print(f"  Original size : {info['original_size']}")
    print(f"  After resize  : {info['after_resize']}")

    print("\n[ STEP 3 ]  DataLoader Batch 8")
    loader8 = get_dataloader(IMAGE_DIR, mode="train", batch_size=8)
    batch, _ = next(iter(loader8))
    print(f"  Batch shape   : {batch.shape}")

    print("\n[ STEP 4 ]  DataLoader Batch 16")
    loader16 = get_dataloader(IMAGE_DIR, mode="train", batch_size=16)
    batch16, _ = next(iter(loader16))
    print(f"  Batch shape   : {batch16.shape}")

    print("\n[ STEP 5 ]  3 Optimizers")
    model = torch.nn.Linear(10, 2)
    opt1 = get_optimizer(model, "sgd")
    opt2 = get_optimizer(model, "adam")
    opt3 = get_optimizer(model, "adamw")
    print(f"  1. {opt1.__class__.__name__}   → simple")
    print(f"  2. {opt2.__class__.__name__}  → recommended ✅")
    print(f"  3. {opt3.__class__.__name__} → large models")

    print("\n" + "="*50)
    print("  ALL DONE!")
    print("="*50)