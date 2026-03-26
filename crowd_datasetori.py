import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io

class CrowdDataset(Dataset):
    def __init__(self, img_path, gt_path):
        self.img_path = img_path
        self.gt_path = gt_path

        self.img_files = sorted(os.listdir(img_path))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_name = self.img_files[index]

        
        # IMAGE PATH
         
        img_file = os.path.join(self.img_path, img_name)

         
        # GT PATH
        
        gt_name = "GT_" + img_name.replace(".jpg", ".mat")
        gt_file = os.path.join(self.gt_path, gt_name)

         
        # LOAD IMAGE
         
        img = Image.open(img_file).convert("RGB")

        # 🔥 FIX: resize image
        img = img.resize((224, 224))

        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

         
        # LOAD GROUND TRUTH
         
        mat = scipy.io.loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

         
        # CREATE DENSITY MAP
         
        density_map = np.zeros((224, 224), dtype=np.float32)

        for p in points:
            x = min(223, max(0, int(p[0] * 224 / img.shape[2])))
            y = min(223, max(0, int(p[1] * 224 / img.shape[1])))
            density_map[y, x] += 1

        # Apply Gaussian Blur
        density_map = cv2.GaussianBlur(density_map, (15, 15), 4)

        # Convert to tensor
        density_map = torch.tensor(density_map).unsqueeze(0)

        return img, density_map, img_name