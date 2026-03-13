import os
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.csrnet import CSRNet


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_folder = "data/part_A_final/test_data/images"
    gt_folder = "data/part_A_final/test_data/ground_truth"

    img_list = sorted(os.listdir(img_folder))

    # Find first image with matching GT
    for img_name in img_list:
        gt_name = "GT_" + img_name.replace(".jpg", ".mat")
        gt_path = os.path.join(gt_folder, gt_name)

        if os.path.exists(gt_path):
            break

    print("Selected Image:", img_name)

    img_path = os.path.join(img_folder, img_name)
 
    #  ORIGINAL IMAGE
    
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    print("Original Size:", orig_w, "x", orig_h)
 
    #  RESIZE TO 224x224
     
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_resized = resize_transform(img)
    resized_tensor = img_resized.unsqueeze(0).to(device)

    print("Resized Tensor Shape:", resized_tensor.shape)
 
    #  LOAD GROUND TRUTH
     
    mat = scipy.io.loadmat(gt_path)
    points = mat["image_info"][0][0][0][0][0]
    gt_count = len(points)
    print("Ground Truth Count:", gt_count)
 
    #  LOAD MODEL
     
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
    model.eval()
    print("Model Loaded Successfully")
 
    #  PREDICT DENSITY MAP
     
    with torch.no_grad():
        output = model(resized_tensor)
        pred_count = output.sum().item()

    print("Predicted Count:", round(pred_count, 2))

    density_map = output.squeeze().cpu().numpy()
 
    #  VISUALIZATION
     
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image ({orig_w}x{orig_h})")
    plt.axis("off")

    # Resized image
    plt.subplot(2, 2, 2)
    plt.imshow(img_resized.permute(1, 2, 0))
    plt.title("Resized Image (224x224)")
    plt.axis("off")

    # Ground Truth points
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],
                s=10, c="red")
    plt.title(f"Ground Truth Count: {gt_count}")
    plt.axis("off")

    # Density Map
    plt.subplot(2, 2, 4)
    plt.imshow(density_map, cmap="jet")
    plt.title(f"Predicted Count: {round(pred_count, 2)}")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()