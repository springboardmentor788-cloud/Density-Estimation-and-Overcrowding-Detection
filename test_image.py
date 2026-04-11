import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.csrnet import CSRNet
from preprocess import load_annotation
import config


# ---------------- LOAD MODEL ---------------- #
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSRNet().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print("✅ Model loaded")

    return model, device


# ---------------- PROCESS IMAGE ---------------- #
def process_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"❌ Cannot load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = img.copy()

    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

    return img, original


# ---------------- MAIN TEST FUNCTION ---------------- #
def test():

    model, device = load_model()

    image_dir = os.path.join(
        config.BASE_DIR,
        "data",
        "ShanghaiTech",
        "part_A",
        "test_data",
        "images"
    )

    gt_dir = os.path.join(
        config.BASE_DIR,
        "data",
        "ShanghaiTech",
        "part_A",
        "test_data",
        "ground-truth"
    )

    image_list = sorted(os.listdir(image_dir))

    # ✅ LIMIT TO 10 SAMPLES
    random.seed(42)  # reproducibility
    image_list = random.sample(image_list, 10)

    print(f"🔍 Testing on {len(image_list)} images\n")

    total_mae = 0

    for img_name in image_list:

        img_path = os.path.join(image_dir, img_name)
        mat_name = "GT_" + img_name.replace(".jpg", ".mat")
        mat_path = os.path.join(gt_dir, mat_name)

        # Load image
        img_tensor, original_img = process_image(img_path)
        img_tensor = img_tensor.to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)

        density_map = output.squeeze().cpu().numpy()

        # ✅ Predicted count
        pred_count = density_map.sum()

        # ✅ Ground truth count
        points = load_annotation(mat_path)
        gt_count = len(points)

        error = abs(pred_count - gt_count)
        total_mae += error

        print(f"{img_name} | Pred: {pred_count:.2f} | GT: {gt_count} | Error: {error:.2f}")

        # ---------------- VISUALIZATION ---------------- #
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(original_img)
        plt.axis("off")

        # Density map
        plt.subplot(1, 3, 2)
        plt.title(f"Density Map\nCount: {pred_count:.2f}")
        plt.imshow(density_map, cmap="jet")
        plt.axis("off")

        # Overlay
        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        overlay = cv2.resize(
            density_map,
            (original_img.shape[1], original_img.shape[0])
        )
        plt.imshow(original_img)
        plt.imshow(overlay, alpha=0.5, cmap="jet")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    avg_mae = total_mae / len(image_list)
    print(f"\n📊 Test MAE (10 samples): {avg_mae:.2f}")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    test()