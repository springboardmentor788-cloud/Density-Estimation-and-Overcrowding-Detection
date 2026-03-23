import torch
from torch.utils.data import DataLoader
from dataset_loader import ShanghaiTechDataset
from models.csrnet import CSRNet
import torch.nn as nn
import torch.optim as optim

# TRAIN DATA
train_img_path = r"dataset/ShanghaiTech/part_A/train_data/images"
train_gt_path = r"dataset/ShanghaiTech/part_A/train_data/ground-truth"

# TEST DATA (ShanghaiTech already provides this)
test_img_path = r"dataset/ShanghaiTech/part_A/test_data/images"
test_gt_path = r"dataset/ShanghaiTech/part_A/test_data/ground-truth"

train_dataset = ShanghaiTechDataset(train_img_path, train_gt_path)
test_dataset = ShanghaiTechDataset(test_img_path, test_gt_path)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

model = CSRNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(50):

    model.train()
    train_loss = 0

    for img, density in train_loader:

        pred = model(img)

        loss = criterion(pred, density)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print("Epoch:", epoch, "Train Loss:", train_loss)


torch.save(model.state_dict(), "crowd_model.pth")
print("Model saved successfully ✅")