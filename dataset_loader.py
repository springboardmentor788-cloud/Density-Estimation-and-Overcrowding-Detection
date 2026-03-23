import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter


class ShanghaiTechDataset(Dataset):

    def __init__(self, img_path, gt_path):

        self.img_path = img_path
        self.gt_path = gt_path
        self.image_files = sorted(os.listdir(img_path))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def generate_density_map(self, shape, points):

        density = np.zeros(shape)

        for p in points:
            x = min(int(p[0]), shape[1]-1)
            y = min(int(p[1]), shape[0]-1)
            density[y, x] += 1

        density = gaussian_filter(density, sigma=4)

        return density

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_file = os.path.join(self.img_path, img_name)

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_name = "GT_" + img_name.replace(".jpg",".mat")
        gt_file = os.path.join(self.gt_path, gt_name)

        mat = sio.loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

        density = self.generate_density_map(img.shape[:2], points)

        img = self.transform(img)
        density = cv2.resize(density, (28,28))
        density = torch.tensor(density).float().unsqueeze(0)
        return img, density