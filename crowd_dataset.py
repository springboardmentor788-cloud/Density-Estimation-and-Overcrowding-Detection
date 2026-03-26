import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io as sio
import cv2

class CrowdDataset(Dataset):
    def __init__(self, root_dir, mode="train_data"):
        self.img_dir = os.path.join(root_dir, mode, "images")
        self.gt_dir = os.path.join(root_dir, mode, "ground_truth")

        self.img_list = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".jpg")
        ])

        print("✅ Loaded images:", len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, "GT_" + img_name.replace(".jpg", ".mat"))

        # ===== IMAGE =====
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        img = img.resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        # ===== LOAD GT =====
        mat = sio.loadmat(gt_path)

        # TRY MULTIPLE FORMATS (AUTO FIX)
        try:
            points = mat["image_info"][0][0][0][0][0]
        except:
            try:
                points = mat["image_info"][0][0][0][0]
            except:
                try:
                    points = mat["image_info"][0][0][0]
                except:
                    print("❌ GT format error:", gt_path)
                    points = []

        # ===== CREATE DENSITY =====
        density = np.zeros((224, 224), dtype=np.float32)

        scale_x = 224 / w
        scale_y = 224 / h

        for p in points:
            x = int(p[0] * scale_x)
            y = int(p[1] * scale_y)

            if x >= 224 or y >= 224:
                continue

            density[y, x] += 1

        #  DEBUG LINE (IMPORTANT)
        # print("GT SUM:", density.sum())

        # Smooth density
        density = cv2.GaussianBlur(density, (15, 15), 0)

        density = torch.tensor(density).unsqueeze(0)

        return img, density, 0