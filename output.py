import torch
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from model import CSRNet

# -------------------------
# PATHS
# -------------------------
image_path = "data/part_A_final/test_data/images/IMG_73.jpg"
gt_path = "data/part_A_final/test_data/ground_truth/GT_IMG_73.mat"

device = torch.device("cpu")

# -------------------------
# LOAD MODEL
# -------------------------
model = CSRNet()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------
# LOAD IMAGE
# -------------------------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_img = img.copy()

img_resized = cv2.resize(img, (224, 224))
img_resized = img_resized / 255.0

# Normalize
img_resized = (img_resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

img_resized = img_resized.transpose(2, 0, 1)
input_img = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).to(device)

# -------------------------
# PREDICTION
# -------------------------
with torch.no_grad():
    output = model(input_img)

pred_density = output.squeeze().cpu().numpy()
predicted_count = output.sum().item()

# Resize density to match original image
pred_density_resized = cv2.resize(pred_density, (orig_img.shape[1], orig_img.shape[0]))

# -------------------------
# GROUND TRUTH
# -------------------------
mat = sio.loadmat(gt_path)
points = mat["image_info"][0][0][0][0][0]
gt_count = len(points)

# Create GT density map
gt_density = np.zeros((orig_img.shape[0], orig_img.shape[1]))

for p in points:
    x, y = int(p[0]), int(p[1])
    if x < orig_img.shape[1] and y < orig_img.shape[0]:
        gt_density[y, x] = 1

gt_density = gaussian_filter(gt_density, sigma=4)

# -------------------------
# ERROR
# -------------------------
abs_error = abs(gt_count - predicted_count)

# -------------------------
# OVERLAY MAPS
# -------------------------
pred_overlay = cv2.applyColorMap((pred_density_resized / pred_density_resized.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
gt_overlay = cv2.applyColorMap((gt_density / gt_density.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)

overlay_combined = cv2.addWeighted(pred_overlay, 0.5, gt_overlay, 0.5, 0)

# -------------------------
# DISPLAY RESULTS
# -------------------------
plt.figure(figsize=(15,10))

# Original Image
plt.subplot(2,3,1)
plt.title("Original Image")
plt.imshow(orig_img)
plt.axis("off")

# GT Density
plt.subplot(2,3,2)
plt.title(f"GT Density (Count={gt_count})")
plt.imshow(gt_density, cmap='jet')
plt.axis("off")

# Predicted Density
plt.subplot(2,3,3)
plt.title(f"Pred Density (Count={predicted_count:.2f})")
plt.imshow(pred_density_resized, cmap='jet')
plt.axis("off")

# Overlay
plt.subplot(2,3,4)
plt.title("Pred + GT Overlay")
plt.imshow(overlay_combined)
plt.axis("off")

# Error Text
plt.subplot(2,3,5)
plt.title("Error Info")
plt.text(0.1, 0.5, f"Absolute Error: {abs_error:.2f}", fontsize=14)
plt.axis("off")

plt.tight_layout()

# -------------------------
# PRINT VALUES (MOVE HERE)
# -------------------------
print("\n========== RESULT ==========")
print(f"Actual Count     : {gt_count}")
print(f"Predicted Count  : {predicted_count:.2f}")
print(f"Absolute Error   : {abs_error:.2f}")


plt.show()