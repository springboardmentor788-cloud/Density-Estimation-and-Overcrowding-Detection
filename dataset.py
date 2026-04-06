import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split


class CrowdDataset(Dataset):

    def __init__(self, root_paths, split="train"):

        # ✅ Allow both string and list
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        self.image_files = []
        self.gt_files = []

        print("🔍 Loading Dataset...")

        # ------------------------
        # Combine multiple datasets
        # ------------------------
        for root_path in root_paths:

            image_path = os.path.join(root_path, "images")
            gt_path = os.path.join(root_path, "ground-truth")

            print("Images Path:", image_path)
            print("GT Path:", gt_path)

            # ✅ Check paths
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"❌ Images folder not found: {image_path}")

            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"❌ Ground truth folder not found: {gt_path}")

            images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]

            if len(images) == 0:
                raise Exception(f"❌ No images found in: {image_path}")

            for img in images:
                self.image_files.append(os.path.join(image_path, img))

                gt_name = "GT_" + img.replace(".jpg", ".mat")
                self.gt_files.append(os.path.join(gt_path, gt_name))

        print(f"✅ Total Images Loaded: {len(self.image_files)}")

        # ------------------------
        # Split dataset (80/10/10)
        # ------------------------
        train_imgs, temp_imgs, train_gts, temp_gts = train_test_split(
            self.image_files, self.gt_files, test_size=0.2, random_state=42
        )

        val_imgs, test_imgs, val_gts, test_gts = train_test_split(
            temp_imgs, temp_gts, test_size=0.5, random_state=42
        )

        if split == "train":
            self.image_files = train_imgs
            self.gt_files = train_gts
        elif split == "val":
            self.image_files = val_imgs
            self.gt_files = val_gts
        else:
            self.image_files = test_imgs
            self.gt_files = test_gts

        print(f"📊 Using {split} split: {len(self.image_files)} samples")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = self.image_files[idx]
        gt_path = self.gt_files[idx]

        # ------------------------
        # Load Image
        # ------------------------
        img = cv2.imread(img_path)

        if img is None:
            raise Exception(f"❌ Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # ------------------------
        # Make divisible by 8
        # ------------------------
        h = h - (h % 8)
        w = w - (w % 8)

        img = img[:h, :w]

        # ------------------------
        # Load Ground Truth
        # ------------------------
        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]

        # ------------------------
        # Create Density Map
        # ------------------------
        density = np.zeros((h, w))

        for point in points:
            x = min(int(point[0]), w - 1)
            y = min(int(point[1]), h - 1)
            density[y, x] = 1

        density = gaussian_filter(density, sigma=15)

        # ------------------------
        # Downsample Density Map
        # ------------------------
        density = cv2.resize(density, (w // 8, h // 8))
        density = density * 64

        # ------------------------
        # Convert to Tensor
        # ------------------------
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        density = torch.from_numpy(density).unsqueeze(0).float()

        return img, density