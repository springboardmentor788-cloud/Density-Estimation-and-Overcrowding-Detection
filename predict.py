import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import CSRNet


# -------------------------
# Load Model
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_model.pth", map_location=device))
model.eval()


# -------------------------
# Load Image
# -------------------------

img_path = "Dataset/ShanghaiTech/part_B/train_data/images/IMG_1.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# same resize used in training
img = cv2.resize(img, (512, 384))

input_img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
input_img = input_img.unsqueeze(0).to(device)


# -------------------------
# Prediction
# -------------------------

with torch.no_grad():
    output = model(input_img)


density_map = output.squeeze().cpu().numpy()

# crowd count
count = density_map.sum()

print("Estimated Crowd Count:", int(count))


# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Predicted Density Map")
plt.imshow(density_map, cmap="jet")

plt.show()