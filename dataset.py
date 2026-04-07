import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

from preprocess import load_image, load_annotation, generate_density_map
import config


# ================= DATASET =================
class CrowdDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.samples = []

        # 🔥 Fix 1: Handle wrong folder naming automatically
        if not os.path.exists(gt_dir):
            alt = gt_dir.replace("ground_truth", "ground-truth")
            if os.path.exists(alt):
                gt_dir = alt

        if not os.path.exists(gt_dir):
            raise Exception(f"❌ Ground truth folder not found: {gt_dir}")

        # 🔥 Fix 2: Handle nested GT folder (ShanghaiTech structure)
        subfolders = os.listdir(gt_dir)
        if len(subfolders) == 1:
            possible = os.path.join(gt_dir, subfolders[0])
            if os.path.isdir(possible):
                gt_dir = possible

        if not os.path.exists(image_dir):
            raise Exception(f"❌ Image folder not found: {image_dir}")

        image_files = sorted(os.listdir(image_dir))

        for img_name in image_files:
            if not img_name.endswith(".jpg"):
                continue

            number = img_name.split("_")[1].split(".")[0]
            gt_name = f"GT_IMG_{number}.mat"

            img_path = os.path.join(image_dir, img_name)
            gt_path = os.path.join(gt_dir, gt_name)

            if os.path.exists(gt_path):
                self.samples.append((img_path, gt_path))

        if len(self.samples) == 0:
            raise Exception("❌ No valid image-GT pairs found!")

        print("✅ Valid samples found:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        image = load_image(img_path)   # HWC
        points = load_annotation(gt_path)

        orig_h, orig_w, _ = image.shape

        # Resize image
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # Scale points
        scale_x = config.IMAGE_SIZE / orig_w
        scale_y = config.IMAGE_SIZE / orig_h

        scaled_points = []
        for p in points:
            x = p[0] * scale_x
            y = p[1] * scale_y

            if x < config.IMAGE_SIZE and y < config.IMAGE_SIZE:
                scaled_points.append([x, y])

        # Normalize image
        image = image / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

        image = np.transpose(image, (2, 0, 1))

        # Density map
        density = generate_density_map(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            scaled_points
        )

        density = cv2.resize(
            density,
            (config.DENSITY_SIZE, config.DENSITY_SIZE)
        )

        # To tensor
        image = torch.tensor(image, dtype=torch.float32)
        density = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        return image, density


# ================= DATALOADER =================
def get_dataloaders():

    print("📂 Loading ShanghaiTech dataset...")

    BASE_PATH = r"C:\Users\DELL\Deep Vision\data\ShanghaiTech\part_A"

    train_img = os.path.join(BASE_PATH, "train_data", "images")
    train_gt = os.path.join(BASE_PATH, "train_data", "ground_truth")  # will auto-fix

    val_img = os.path.join(BASE_PATH, "test_data", "images")
    val_gt = os.path.join(BASE_PATH, "test_data", "ground_truth")    # will auto-fix

    # Create datasets
    train_dataset = CrowdDataset(train_img, train_gt)
    val_dataset = CrowdDataset(val_img, val_gt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Windows safe
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    print("✅ DataLoaders ready")
    return train_loader, val_loader