import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import CSRNet
import random

# -------------------------
# LOSS FUNCTION
# -------------------------
mse_loss = nn.MSELoss()

def combined_loss(pred, gt):
    mse = mse_loss(pred, gt)
    count_loss = torch.abs(
        pred.sum(dim=[1,2,3]) - gt.sum(dim=[1,2,3])
    ).mean()
    return mse + 0.1 * count_loss


# -------------------------
# DATASET
# -------------------------
class CrowdDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_file = self.image_paths[idx]

        gt_file = img_file.replace("images", "ground_truth") \
                          .replace(".jpg", ".mat") \
                          .replace("IMG_", "GT_IMG_")

        img = cv2.imread(img_file)

        if img is None or not os.path.exists(gt_file):
            return torch.zeros(3,224,224), torch.zeros(1,28,28)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Resize + Normalize
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2,0,1)
        img = torch.tensor(img, dtype=torch.float32)

        # Load GT
        mat = sio.loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

        # Create density map
        density = np.zeros((224,224), dtype=np.float32)

        for p in points:
            x = int(p[0] * 224 / w)
            y = int(p[1] * 224 / h)

            if 0 <= x < 224 and 0 <= y < 224:
                density[y, x] = 1

        # Gaussian
        density = gaussian_filter(density, sigma=4)

        # 🔥 Normalize BEFORE resize
        if density.sum() != 0:
            density = density / density.sum() * len(points)

        # Downsample
        density = cv2.resize(density, (28,28))

        # Augmentation (flip)
        if random.random() > 0.5:
            img = np.flip(img.numpy(), axis=2).copy()
            density = np.flip(density, axis=1).copy()
            img = torch.tensor(img, dtype=torch.float32)

        density = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        return img, density


# -------------------------
# LOAD DATA
# -------------------------
def load_all_images():
    paths = [
        "data/part_A_final/train_data/images",
        "data/part_B_final/train_data/images"
    ]

    image_paths = []
    for p in paths:
        for f in os.listdir(p):
            if f.endswith(".jpg"):
                image_paths.append(os.path.join(p, f))

    return image_paths


all_images = load_all_images()
random.shuffle(all_images)

train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

train_loader = DataLoader(CrowdDataset(train_imgs), batch_size=8, shuffle=True)
val_loader = DataLoader(CrowdDataset(val_imgs), batch_size=8, shuffle=False)


# -------------------------
# MODEL
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# -------------------------
# TRAINING
# -------------------------
epochs = 15
best_val_loss = float('inf')

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images, gt in train_loader:

        images = images.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        output = model(images)

        loss = combined_loss(output, gt)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0
    mae = 0

    with torch.no_grad():
        for images, gt in val_loader:

            images = images.to(device)
            gt = gt.to(device)

            output = model(images)

            val_loss += combined_loss(output, gt).item()

            for i in range(images.size(0)):
                pred_count = output[i].sum().item()
                gt_count = gt[i].sum().item()
                mae += abs(pred_count - gt_count)

    val_loss /= len(val_loader)
    mae /= len(val_loader.dataset)

    print(f"\nEpoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f} | MAE {mae:.2f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")


# -------------------------
# SAVE FINAL
# -------------------------
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/model.pth")
torch.save(model.state_dict(), "final_model.pth")

print("🎉 Training complete!")