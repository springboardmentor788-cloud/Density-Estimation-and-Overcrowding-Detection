
import torch
from torch.utils.data import DataLoader
from dataset import ShanghaiDataset
from models.csrnet import CSRNet

# Path to Part A
dataset_path = "data/part_A_final"

# Load dataset
dataset = ShanghaiDataset(dataset_path, train=True)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model
model = CSRNet()

for img, gt in loader:
    output = model(img)
    print("Image shape:", img.shape)
    print("GT shape:", gt.shape)
    print("Output shape:", output.shape)
    break