import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.csrnet import CSRNet
from crowd_dataset import CrowdDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

 
# FOLDERS
 
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

 
# DEVICE
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

 
# DATASET
 
dataset_A = CrowdDataset("data/part_A_final", "train_data")
dataset_B = CrowdDataset("data/part_B_final", "train_data")

full_dataset = ConcatDataset([dataset_A, dataset_B])

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_data, val_data, _ = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2)

 
# MODEL
 
model = CSRNet().to(device)

#  CORRECT PATH HERE
pretrained_path = "models/pretrained_csrnet.pth"

if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    print("✅ Pretrained weights loaded successfully")
else:
    print(" Pretrained weights not found → training from scratch")


 
# LOSS & OPTIMIZER

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

 
# TRAINING
 
epochs = 30
train_losses = []
val_losses = []

print("\n Training Started...\n")

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for img, gt, _ in train_loader:
        img, gt = img.to(device), gt.to(device)

        pred = model(img)

        gt = F.interpolate(gt, size=pred.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    
    # VALIDATION
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, gt, _ in val_loader:
            img, gt = img.to(device), gt.to(device)

            pred = model(img)
            gt = F.interpolate(gt, size=pred.shape[2:], mode='bilinear', align_corners=False)

            val_loss += criterion(pred, gt).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

 
# SAVE MODEL
 
save_path = "models/csrnet_main.pth"
torch.save(model.state_dict(), save_path)

print("\n✅ Model Saved at:", save_path)

 
# GRAPH
 
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()

plt.savefig("outputs/graphs/loss.png")
plt.show()

print(" Graph saved at outputs/graphs/loss.png")