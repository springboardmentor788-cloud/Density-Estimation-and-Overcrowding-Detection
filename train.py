import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CrowdDataset
from model import CSRNet


# -----------------------------
# Dataset Load
# -----------------------------

dataset = CrowdDataset("Dataset/ShanghaiTech/part_B/train_data")

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


# -----------------------------
# Model
# -----------------------------

model = CSRNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze VGG frontend
for param in model.frontend.parameters():
    param.requires_grad = False


# -----------------------------
# Loss and Optimizer
# -----------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)


# -----------------------------
# Training Settings
# -----------------------------

epochs = 50

print("Training Started...")

for epoch in range(epochs):

    # Unfreeze frontend after 10 epochs
    if epoch == 10:
        print("Unfreezing frontend layers...")
        for param in model.frontend.parameters():
            param.requires_grad = True

    epoch_loss = 0

    for images, densities in dataloader:

        images = images.to(device)
        densities = densities.to(device)

        outputs = model(images)

        loss = criterion(outputs, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss}")


# -----------------------------
# Save Model
# -----------------------------

torch.save(model.state_dict(), "csrnet_model.pth")
print("Model saved as csrnet_model.pth")