import torch
from torch.utils.data import DataLoader, random_split
from data_loader import CrowdDataset
from csrnet import CSRNet
import torch.nn as nn
import matplotlib.pyplot as plt
import os

torch.set_num_threads(2)
os.makedirs("Results", exist_ok=True)

# DATA
image_path = "dataset/ShanghaiTech/part_A/train_data/images"
gt_path = "dataset/ShanghaiTech/part_A/train_data/ground-truth"

dataset = CrowdDataset(image_path, gt_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# MODEL
model = CSRNet()

# 🔥 PARTIAL TRAINING (BEST BALANCE)
for name, param in model.named_parameters():
    if "backend" in name or "output_layer" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

EPOCHS = 12

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    # TRAIN
    model.train()
    train_loss = 0

    for img, gt in train_loader:
        out = model(img)
        loss = criterion(out, gt)

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
        for img, gt in val_loader:
            out = model(img)
            val_loss += criterion(out, gt).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "csrnet.pth")

# GRAPH
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("Results/loss_graph.png")
plt.close()

print("✅ Training complete + graph saved")