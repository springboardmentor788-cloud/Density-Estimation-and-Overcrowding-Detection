import torch
import numpy as np
from models.csrnet import CSRNet
from crowd_datasetori import CrowdDataset
from torch.utils.data import DataLoader

 
# DEVICE
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
# LOAD MODEL
 
model = CSRNet().to(device)
model.load_state_dict(torch.load("models/csrnet_main.pth", map_location=device))
model.eval()

print("✅ Model Loaded")

 
# DATASET (USE TRAIN DATA FOR NOW)
 
A_img = "data/part_A_final/train_data/images"
A_gt  = "data/part_A_final/train_data/ground_truth"

B_img = "data/part_B_final/train_data/images"
B_gt  = "data/part_B_final/train_data/ground_truth"

dataset_A = CrowdDataset(A_img, A_gt)
dataset_B = CrowdDataset(B_img, B_gt)

full_dataset = dataset_A + dataset_B

loader = DataLoader(full_dataset, batch_size=1)

print("✅ Total images:", len(full_dataset))

 
# METRICS
 
mae = 0
mse = 0

 
# EVALUATION
 
with torch.no_grad():
    for img, gt, _ in loader:

        img = img.to(device)
        gt = gt.to(device)

        output = model(img)

        pred_count = output.sum().item()
        gt_count = gt.sum().item()

        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count) ** 2

 
# FINAL RESULTS

mae = mae / len(loader)
mse = (mse / len(loader)) ** 0.5

print("\n FINAL RESULTS")
print("MAE:", round(mae, 2))
print("RMSE:", round(mse, 2))