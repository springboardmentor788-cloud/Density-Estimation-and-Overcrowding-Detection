import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# ----------------------------
# Dataset Class
# ----------------------------

class CrowdDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_path = os.path.join(root_path, "images")
        self.gt_path = os.path.join(root_path, "ground-truth")

        self.image_files = [f for f in os.listdir(self.image_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_path, img_name)

        gt_name = "GT_" + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_path, gt_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]

        density = np.zeros((h, w))
        for point in points:
            x = min(int(point[0]), w - 1)
            y = min(int(point[1]), h - 1)
            density[y, x] = 1

        density = gaussian_filter(density, sigma=15)

        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        density = torch.from_numpy(density).unsqueeze(0).float()

        return img, density

# ----------------------------
# Simple CNN Model
# ----------------------------

class SimpleCrowdNet(nn.Module):
    def __init__(self):
        super(SimpleCrowdNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,1,1)
        )

    def forward(self, x):
        return self.layers(x)

# ----------------------------
# Training Setup
# ----------------------------

dataset = CrowdDataset("Dataset/ShanghaiTech/part_B/train_data")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = SimpleCrowdNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Training Started...")

for images, densities in dataloader:
    print(images.shape,densities.shape)
    outputs = model(images)
    loss = criterion(outputs, densities)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
    break   # only 1 batch test