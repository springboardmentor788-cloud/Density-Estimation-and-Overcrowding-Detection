import torch
from torch.utils.data import DataLoader
from data_loader import CrowdDataset
from csrnet import CSRNet
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("Results", exist_ok=True)

image_path = "dataset/ShanghaiTech/part_A/test_data/images"
gt_path = "dataset/ShanghaiTech/part_A/test_data/ground-truth"

dataset = CrowdDataset(image_path, gt_path)
loader = DataLoader(dataset, batch_size=1)

model = CSRNet()
model.load_state_dict(torch.load("csrnet.pth"))
model.eval()

mae = 0
preds = []
gts = []

with torch.no_grad():
    for i, (img, gt) in enumerate(loader):

        out = model(img)

        pred = out.sum().item()
        gt_val = gt.sum().item()

        mae += abs(pred - gt_val)

        preds.append(pred)
        gts.append(gt_val)

        # SAVE 5 IMAGES
        if i < 5:
            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.imshow(img.squeeze().permute(1,2,0))
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(gt.squeeze(), cmap='jet')

            plt.subplot(1,3,3)
            plt.imshow(out.squeeze(), cmap='jet')

            plt.savefig(f"Results/sample_{i}.png")
            plt.close()

mae /= len(dataset)

preds = np.array(preds)
gts = np.array(gts)

ss_res = np.sum((gts - preds)**2)
ss_tot = np.sum((gts - np.mean(gts))**2)

r2 = 1 - (ss_res / ss_tot)

print("MAE:", mae)
print("R²:", r2)
print("✅ Results saved")