import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.csrnet import CSRNet   

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
 
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

image_path = "data/part_A_final/test_data/images/IMG_22.jpg"  # Change if needed

img = Image.open(image_path).convert("RGB")

# Resize to 244x244 (as mentor said)
img = img.resize((244, 244))

img = np.array(img)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # Convert HWC → CHW

img = torch.tensor(img).unsqueeze(0).to(device)

print("Input Image Shape:", img.shape)

with torch.no_grad():
    output = model(img)

print("Model Output Shape:", output.shape)

predicted_count = output.sum().item()
print("Predicted Crowd Count:", round(predicted_count, 2))

density_map = output.squeeze().cpu().numpy()

plt.imshow(density_map, cmap='jet')
plt.colorbar()
plt.title("Density Map")
plt.show()