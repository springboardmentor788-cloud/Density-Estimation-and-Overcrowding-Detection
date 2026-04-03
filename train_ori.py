import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.csrnet import CSRNet
from crowd_datasetori import CrowdDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

 
# OUTPUT FOLDERS
 
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

 
# DEVICE
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

 
# DATA PATHS
 
A_img = "data/part_A_final/train_data/images"
A_gt = "data/part_A_final/train_data/ground_truth"

B_img = "data/part_B_final/train_data/images"
B_gt = "data/part_B_final/train_data/ground_truth"

dataset_A = CrowdDataset(A_img, A_gt)
dataset_B = CrowdDataset(B_img, B_gt)

full_dataset = ConcatDataset([dataset_A, dataset_B])

print("Total dataset:", len(full_dataset))

 
# SPLIT
 
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_data, val_data, test_data = random_split(
    full_dataset, [train_size, val_size, test_size]
)

print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))

 
# DATALOADER
 
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

 
# MODEL

model = CSRNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

 
# TRAINING
 
num_epochs = 50    

train_losses = []
val_losses = []

print("Starting Training...")

for epoch in range(num_epochs):

    model.train()
    running_loss = 0

    for img, gt, _ in train_loader:

        img = img.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(img)

        if output.shape != gt.shape:
            gt = F.interpolate(
                gt,
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        loss = criterion(output, gt)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, gt, _ in val_loader:

            img = img.to(device)
            gt = gt.to(device)

            output = model(img)

            if output.shape != gt.shape:
                gt = F.interpolate(
                    gt,
                    size=output.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            loss = criterion(output, gt)
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

 
# SAVE MODEL
 
torch.save(model.state_dict(), "csrnet_fast.pth")

 
# SAVE GRAPH
 
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("outputs/graphs/loss_fast.png")
plt.show()