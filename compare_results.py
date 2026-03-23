import cv2
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision import transforms
from models.csrnet import CSRNet
from scipy.ndimage import gaussian_filter

# 🔹 Paths
img_path = r"dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg"
gt_path = r"dataset/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat"

# 🔹 Load Model
model = CSRNet()
model.load_state_dict(torch.load("crowd_model.pth"))
model.eval()

# 🔹 Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 Load Image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_img = transform(img_rgb).unsqueeze(0)

# 🔹 Load Ground Truth
mat = sio.loadmat(gt_path)
points = mat["image_info"][0][0][0][0][0]

# 🔹 Generate Ground Truth Density
def generate_density(shape, points):
    density = np.zeros(shape)
    for p in points:
        x = min(int(p[0]), shape[1]-1)
        y = min(int(p[1]), shape[0]-1)
        density[y, x] += 1
    density = gaussian_filter(density, sigma=4)
    return density

gt_density = generate_density(img.shape[:2], points)

# 🔹 Predict
with torch.no_grad():
    pred = model(input_img)

pred_density = pred.squeeze().numpy()

# 🔹 Resize prediction to original size
pred_density_resized = cv2.resize(pred_density, (img.shape[1], img.shape[0]))

scale_factor = (img.shape[0] * img.shape[1]) / (28 * 28)

pred_count = pred_density.sum() * scale_factor

# 🔹 Counts
gt_count = gt_density.sum()

scale_factor = (img.shape[0] * img.shape[1]) / (28 * 28)
pred_count = pred_density.sum() * scale_factor

print("Ground Truth:", int(gt_count))
print("Predicted:", int(pred_count))
# 🔹 Plot
plt.figure(figsize=(12,6))

# Input Image
plt.subplot(1,3,1)
plt.title("Input Image")
plt.imshow(img_rgb)
plt.axis("off")

# Ground Truth
plt.subplot(1,3,2)
plt.title(f"Ground Truth\nCount: {gt_count}")
plt.imshow(gt_density, cmap='jet')
plt.axis("off")

# Prediction
plt.subplot(1,3,3)
plt.title(f"Predicted\nCount: {pred_count}")
plt.imshow(pred_density_resized, cmap='jet')
plt.axis("off")

plt.tight_layout()
plt.show()