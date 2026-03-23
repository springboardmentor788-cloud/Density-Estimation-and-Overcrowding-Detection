import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.csrnet import CSRNet

# Load model (no pretrained weights)
model = CSRNet()
model.eval()

print("Model initialized successfully ✅")

# Load image
img_path = "dataset/part_A/test_data/images/IMG_1.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

original_shape = img.shape
print("Original Image Shape:", original_shape)

# Resize to smaller size for CPU speed
img_resized = cv2.resize(img, (224, 224))
print("Resized Image Shape:", img_resized.shape)

# Convert to tensor
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255

# Forward pass
with torch.no_grad():
    density_map = model(img_tensor)

density_map = density_map.squeeze().numpy()

# Calculate count
count = np.sum(density_map)

print("Estimated Crowd Count:", float(count))

# Display results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_resized)
plt.title("Resized Image")
plt.axis("off")

plt.subplot(1,2,2)
density_map_resized = cv2.resize(density_map, (224, 224))
plt.imshow(density_map_resized, cmap="jet")
plt.title("Density Map")
plt.axis("off")

plt.show()