import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.ndimage import gaussian_filter


class CrowdDataset(Dataset):

    def __init__(self, root_path):
        self.root_path = root_path
        self.image_path = os.path.join(root_path, "images")
        self.gt_path = os.path.join(root_path, "ground-truth")

        self.image_files = [f for f in os.listdir(self.image_path) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_path, img_name)

        gt_name = "GT_" + img_name.replace(".jpg", ".mat")
        gt_path = os.path.join(self.gt_path, gt_name)

        # ------------------------
        # Load Image
        # ------------------------

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image (speed up training)
        img = cv2.resize(img, (224, 224))

        h, w = img.shape[:2]
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
        # Resize Density Map
        # (CSRNet output size)
        # ------------------------

        density = cv2.resize(density, (w // 8, h // 8))

        # density normalization
        density = density * 64

        # ------------------------
        # Convert to Tensor
        # ------------------------

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        density = torch.from_numpy(density).unsqueeze(0).float()

        return img, density