import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from scipy.io import loadmat

class CrowdDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:80]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        gt_path = os.path.join(self.gt_dir, "GT_" + img_name.replace('.jpg', '.mat'))

        # IMAGE
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        # GT
        mat = loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]

        density_map = np.zeros((256,256))

        for p in points:
            x = min(int(p[0]), 255)
            y = min(int(p[1]), 255)
            density_map[y][x] += 1

        # 🔥 KEY FIX: strong gaussian smoothing
        density_map = cv2.GaussianBlur(density_map, (25,25), 7)

        # Preserve count
        count = density_map.sum()

        # Resize
        density_map = cv2.resize(density_map, (32,32))

        # Normalize back
        if density_map.sum() != 0:
            density_map = density_map * (count / density_map.sum())

        density_map = torch.tensor(density_map).unsqueeze(0).float()

        return img, density_map