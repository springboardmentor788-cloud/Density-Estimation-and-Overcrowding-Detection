import torch
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import glob

from csrnet import CSRNet
from density_utils import generate_density_map


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path (best checkpoint)
MODEL_PATH = "outputs/epoch_12.pth"

# Test image (for visualization)
IMAGE_PATH = "data/ShanghaiTech/part_A/test_data/images/IMG_1.jpg"


# -------------------------------
# 🔹 Single Image Inference
# -------------------------------
def run_single_image():

    model = CSRNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Model loaded successfully")

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Image not found")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize (keep same as training)
    img_resized = cv2.resize(img_rgb, (224, 224))

    input_tensor = torch.tensor(img_resized)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float() / 255

    input_tensor = input_tensor.to(device)

    # Prediction
    with torch.no_grad():
        pred = model(input_tensor)

    pred_map = pred.squeeze().cpu().numpy()
    pred_count = np.sum(np.maximum(pred_map, 0))

    # -------------------------------
    # Ground Truth
    # -------------------------------
    mat_path = IMAGE_PATH.replace("images", "ground-truth")
    mat_path = mat_path.replace(".jpg", ".mat")
    mat_path = mat_path.replace("IMG_", "GT_IMG_")

    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]

    gt_count = len(points)

    # Generate GT density (correct)
    gt_density = generate_density_map(img.shape[:2], points)
    gt_density = cv2.resize(gt_density, (28, 28))

    # -------------------------------
    # Visualization
    # -------------------------------
    plt.figure(figsize=(14, 6))

    # Input
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Input")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 4, 2)
    plt.imshow(gt_density, cmap="jet")
    plt.title("Ground Truth")
    plt.axis("off")

    # Predicted
    plt.subplot(1, 4, 3)
    plt.imshow(pred_map, cmap="jet")
    plt.title("Predicted")
    plt.axis("off")

    # Results
    plt.subplot(1, 4, 4)
    plt.text(0.1, 0.6, f"GT: {gt_count}", fontsize=14)
    plt.text(0.1, 0.4, f"Pred: {int(pred_count)}", fontsize=14)
    plt.axis("off")
    plt.title("Results")

    plt.tight_layout()
    plt.show()

    print("Ground Truth Count:", gt_count)
    print("Predicted Count:", int(pred_count))


# -------------------------------
# 🔹 MAE Calculation
# -------------------------------
def calculate_mae():

    model = CSRNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image_paths = glob.glob("data/ShanghaiTech/part_A/test_data/images/*.jpg")

    total_error = 0

    for img_path in image_paths:

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        input_tensor = torch.tensor(img)\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float() / 255

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        pred_count = np.sum(np.maximum(pred.squeeze().cpu().numpy(), 0))

        # Ground truth
        mat_path = img_path.replace("images", "ground-truth")
        mat_path = mat_path.replace(".jpg", ".mat")
        mat_path = mat_path.replace("IMG_", "GT_IMG_")

        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]

        gt_count = len(points)

        total_error += abs(pred_count - gt_count)

    mae = total_error / len(image_paths)

    print("\n==============================")
    print("MAE (Mean Absolute Error):", round(mae, 2))
    print("==============================")


# -------------------------------
# 🔹 MAIN
# -------------------------------
if __name__ == "__main__":

    # Show one sample result
    run_single_image()

    # Calculate MAE on full test set
    calculate_mae()