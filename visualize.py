import torch
import matplotlib.pyplot as plt
from csrnet import CSRNet
from data_loader import CrowdDataset

# LOAD DATA
dataset = CrowdDataset(
    "dataset/ShanghaiTech/part_A/train_data/images",
    "dataset/ShanghaiTech/part_A/train_data/ground-truth"
)

img, gt = dataset[0]

# LOAD MODEL
model = CSRNet()
model.load_state_dict(torch.load("csrnet.pth"))
model.eval()

with torch.no_grad():
    output = model(img.unsqueeze(0))

# ----------------------------
# ORIGINAL IMAGE
# ----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img.permute(1,2,0))
plt.title("Original Image")
plt.axis("off")

# ----------------------------
# GROUND TRUTH DENSITY
# ----------------------------
plt.subplot(1,3,2)
plt.imshow(gt.squeeze(), cmap='jet')
plt.title("Ground Truth Density")
plt.colorbar()

# ----------------------------
# PREDICTED DENSITY
# ----------------------------
plt.subplot(1,3,3)
plt.imshow(output.squeeze(), cmap='jet')
plt.title("Predicted Density")
plt.colorbar()

plt.show()

# ----------------------------
# COUNTS
# ----------------------------
pred_count = output.sum().item()
gt_count = gt.sum().item()

print("Predicted Count:", int(pred_count))
print("Actual Count:", int(gt_count))