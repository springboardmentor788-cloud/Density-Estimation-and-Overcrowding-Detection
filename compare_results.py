import os
import cv2   # type: ignore
import torch # type: ignore
import numpy as np # type: ignore
import scipy.io as sio # type: ignore
import matplotlib.pyplot as plt # type: ignore
from torchvision import transforms  # type: ignore
from models.csrnet import CSRNet   # type: ignore
from scipy.ndimage import gaussian_filter   # type: ignore

# 🔹 Paths
img_dir = r"dataset/ShanghaiTech/part_A/test_data/images"
gt_dir = r"dataset/ShanghaiTech/part_A/test_data/ground-truth"

# 🔹 Load Model
model = CSRNet()
model.load_state_dict(torch.load("crowd_model.pth"))
model.eval()

# 🔹 Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 Density generator
def generate_density(shape, points):
    density = np.zeros(shape)
    for p in points:
        x = min(int(p[0]), shape[1]-1)
        y = min(int(p[1]), shape[0]-1)
        density[y, x] += 1
    density = gaussian_filter(density, sigma=4)
    return density

# 🔹 Select images
image_files = sorted(os.listdir(img_dir))[:5]   # change number if needed

plt.figure(figsize=(12, 10))

for i, img_name in enumerate(image_files):

    # ---- Load Image ----
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_img = transform(img_rgb).unsqueeze(0)

    # ---- Ground Truth ----
    gt_name = "GT_" + img_name.replace(".jpg", ".mat")
    gt_path = os.path.join(gt_dir, gt_name)

    mat = sio.loadmat(gt_path)
    points = mat["image_info"][0][0][0][0][0]

    gt_density = generate_density(img.shape[:2], points)

    # ---- Prediction ----
    with torch.no_grad():
        pred = model(input_img)

    pred_density = pred.squeeze().numpy()
    pred_resized = cv2.resize(pred_density, (img.shape[1], img.shape[0]))

    # ---- Counts (FIXED) ----
    gt_count = int(gt_density.sum())

    scale_factor = (img.shape[0]*img.shape[1])/(28*28)
    pred_count = int(pred_density.sum() * scale_factor)

    # ---- Plot ----
    plt.subplot(len(image_files), 3, i*3 + 1)
    plt.imshow(img_rgb)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(len(image_files), 3, i*3 + 2)
    plt.imshow(gt_density, cmap='jet')
    plt.title(f"GT: {gt_count}")
    plt.axis("off")

    plt.subplot(len(image_files), 3, i*3 + 3)
    plt.imshow(pred_resized, cmap='jet')
    plt.title(f"Pred: {pred_count}")
    plt.axis("off")

plt.tight_layout()
plt.show()