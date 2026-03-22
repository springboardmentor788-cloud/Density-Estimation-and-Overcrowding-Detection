import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_loader import CrowdDataset
from csrnet import CSRNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("outputs", exist_ok=True)

# Load dataset
partA = glob.glob("data/ShanghaiTech/part_A/train_data/images/*.jpg")
partB = glob.glob("data/ShanghaiTech/part_B/train_data/images/*.jpg")

image_paths = partA + partB
dataset = CrowdDataset(image_paths)

# Split
total = len(dataset)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size - val_size

train_set, val_set, _ = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

model = CSRNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

epochs = 20

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images, density, _ in train_loader:

        images = images.to(device)
        density = density.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, density)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, density, _ in val_loader:

            images = images.to(device)
            density = density.to(device)

            output = model(images)
            loss = criterion(output, density)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

    torch.save(model.state_dict(), f"outputs/epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "outputs/final_model.pth")

print("Training Complete")