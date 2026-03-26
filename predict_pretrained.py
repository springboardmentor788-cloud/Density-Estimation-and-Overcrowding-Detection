import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.csrnet import CSRNet

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("pretrained_results", exist_ok=True)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
model = CSRNet().to(device)

state_dict = torch.load("models/pretrained_csrnet.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)

model.eval()
print("✅ Pretrained model loaded!")

# =========================
# IMAGE FOLDER
# =========================
img_folder = "data/part_A_final/test_data/images"
img_list = sorted(os.listdir(img_folder))

print("\nProcessing images...\n")

# =========================
# LOOP THROUGH IMAGES
# =========================
for img_name in img_list[:10]:   # first 10 images

    img_path = os.path.join(img_folder, img_name)

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    img = Image.open(img_path).convert("RGB")

    # CSRNet works better with divisible size
    width, height = img.size
    new_w = (width // 8) * 8
    new_h = (height // 8) * 8
    img = img.resize((new_w, new_h))

    img_array = np.array(img).astype(np.float32) / 255.0

    # 🔥 CORRECT NORMALIZATION (VERY IMPORTANT)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_array[:, :, 0] = (img_array[:, :, 0] - mean[0]) / std[0]
    img_array[:, :, 1] = (img_array[:, :, 1] - mean[1]) / std[1]
    img_array[:, :, 2] = (img_array[:, :, 2] - mean[2]) / std[2]

    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array).unsqueeze(0).to(device)

    # -------------------------
    # PREDICTION
    # -------------------------
    with torch.no_grad():
        output = model(img_tensor)

    pred_count = output.sum().item()
    density_map = output.squeeze().cpu().numpy()

    print(f"{img_name} → Count: {round(pred_count, 2)}")

    # -------------------------
    # SAVE RESULT
    # -------------------------
    plt.figure(figsize=(6, 4))

    plt.imshow(density_map, cmap="jet")
    plt.title(f"Predicted Count: {round(pred_count,2)}")
    plt.colorbar()
    plt.axis("off")

    save_path = f"pretrained_results/{img_name}_result.png"
    plt.savefig(save_path)
    plt.close()

print("\n✅ All results saved in 'pretrained_results/'")