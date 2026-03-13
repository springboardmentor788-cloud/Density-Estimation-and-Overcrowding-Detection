import os
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


class ShanghaiDataset(Dataset):
    def __init__(self, root_path, train=True):
        self.train = train

        if train:
            self.image_path = os.path.join(root_path, "train_data", "images")
            self.gt_path = os.path.join(root_path, "train_data", "ground_truth")
        else:
            self.image_path = os.path.join(root_path, "test_data", "images")
            self.gt_path = os.path.join(root_path, "test_data", "ground_truth")

        self.image_files = sorted(glob.glob(os.path.join(self.image_path, "*.jpg")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        # Ground truth path
        file_name = os.path.basename(img_path).replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
        gt_file = os.path.join(self.gt_path, file_name)

        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"GT file not found: {gt_file}")

        # Load .mat file correctly
        mat = loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

        # Generate density map
        h, w = img.shape[1], img.shape[2]
        density = np.zeros((h, w), dtype=np.float32)
        print(h,w)

        for point in points:
            x = min(w - 1, max(0, int(point[0])))
            y = min(h - 1, max(0, int(point[1])))
            density[y, x] = 1

        density = gaussian_filter(density, sigma=4)
        density = np.expand_dims(density, 0)
        density = torch.tensor(density)

        return img, density


def get_dataloader(root_path, train=True, batch_size=1):
    dataset = ShanghaiDataset(root_path, train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)