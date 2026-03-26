import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.csrnet import CSRNet

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

# -----------------------------
# IMAGE PATH
# -----------------------------
image_path = "data/part_A_final/test_data/images/IMG_2.jpg"

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = Image.open(image_path).convert("RGB")

# ⚠️ IMPORTANT: make divisible by 8
img = img.resize((244, 244))    

img_np = np.array(img).astype(np.float32) / 255.0
img_np = np.transpose(img_np, (2, 0, 1))

img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

print("Input Image Shape:", img_tensor.shape)

# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():
    output = model(img_tensor)

print("Model Output Shape:", output.shape)

predicted_count = output.sum().item()
print("Predicted Crowd Count:", round(predicted_count, 2))

density_map = output.squeeze().cpu().numpy()

# -----------------------------
# SAVE OUTPUT IMAGE
# -----------------------------
save_path = "outputs/predictions/prediction_result.png"

plt.figure(figsize=(6,5))
plt.imshow(density_map, cmap='jet')
plt.colorbar()
plt.title(f"Density Map | Count: {round(predicted_count,2)}")

plt.savefig(save_path)
plt.close()

print("Saved at:", save_path)

# -----------------------------
# SHOW (OPTIONAL)
# -----------------------------
plt.imshow(density_map, cmap='jet')
plt.title("Density Map")
plt.show()