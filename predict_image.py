import cv2
import torch
import numpy as np
from torchvision import transforms
from models.csrnet import CSRNet
import matplotlib.pyplot as plt

# 🔹 Load Model
model = CSRNet()
model.load_state_dict(torch.load("crowd_model.pth"))
model.eval()

# 🔹 Image Path (change this)
img_path = r"dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg"

# 🔹 Transform (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 Load Image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_img = transform(img_rgb).unsqueeze(0)

# 🔹 Predict
with torch.no_grad():
    output = model(input_img)

# 🔹 Convert density map
density_map = output.squeeze().numpy()

# 🔹 Get Count
count = density_map.sum()

print("Predicted Crowd Count:", int(count))

# 🔹 Resize heatmap to original image size
heatmap = cv2.resize(density_map, (img.shape[1], img.shape[0]))

# 🔹 Normalize heatmap
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)

# 🔹 Apply color map
heatmap = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)

# 🔹 Overlay heatmap on image
overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# 🔹 Show results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Density Map (Raw)")
plt.imshow(density_map, cmap='jet')
plt.colorbar()
plt.axis("off")

plt.show()