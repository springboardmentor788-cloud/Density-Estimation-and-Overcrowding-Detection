import os
import torch
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CrowdDataset(Dataset):
    def __init__(self, img_path, gt_path):
        self.img_path = img_path
        self.gt_path = gt_path
        self.img_files = sorted(os.listdir(img_path))

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_name = self.img_files[idx]
        img_file = os.path.join(self.img_path, img_name)

        gt_name = "GT_" + img_name.replace(".jpg", ".mat")
        gt_file = os.path.join(self.gt_path, gt_name)

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        img = Image.open(img_file).convert("RGB")
        w, h = img.size

        # -------------------------
        # SMART RESIZE (HALF SIZE)
        # -------------------------
        new_w = (w // 2)
        new_h = (h // 2)

        # make divisible by 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        img = img.resize((new_w, new_h))
        img = self.transform(img)

        # -------------------------
        # LOAD GT
        # -------------------------
        mat = scipy.io.loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

        # -------------------------
        # CREATE DENSITY MAP
        # -------------------------
        density = np.zeros((new_h, new_w))

        scale_x = new_w / w
        scale_y = new_h / h

        for p in points:
            x = min(int(p[0] * scale_x), new_w - 1)
            y = min(int(p[1] * scale_y), new_h - 1)
            density[y, x] += 1

        density = torch.from_numpy(density).float().unsqueeze(0)

        return img, density, img_name