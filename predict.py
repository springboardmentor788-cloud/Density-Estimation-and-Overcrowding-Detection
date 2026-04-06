import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CSRNet


# -------------------------
# Load Model (Reusable)
# -------------------------

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("csrnet_model.pth", map_location=device))
    model.eval()
    return model, device


# -------------------------
# 🔥 CORE FUNCTION (NEW)
# -------------------------

def predict_count(frame, model, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (512, 384))   # SAME as training

    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame)

    density_map = output.squeeze().cpu().numpy()
    count = np.sum(density_map)

    return count, density_map


# -------------------------
# IMAGE PREDICTION (UPDATED)
# -------------------------

def predict_image(img_path):

    model, device = load_model()

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 🔥 USE SAME FUNCTION (reuse)
    count, density_map = predict_count(img, model, device)

    print("Estimated Crowd Count:", int(count))

    # -------------------------
    # Visualization
    # -------------------------

    plt.figure(figsize=(12, 5))

    # Input Image
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_rgb)

    # Density Map
    plt.subplot(1, 2, 2)
    plt.title(f"Density Map | Count: {count:.2f}")
    plt.imshow(density_map, cmap="jet")
    plt.colorbar()

    # Save
    save_path = f"outputs/density_map_{int(count)}.png"
    plt.savefig(save_path)
    print(f"Output saved at: {save_path}")

    plt.show()
    plt.close()