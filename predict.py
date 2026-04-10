import torch
import cv2
import numpy as np

from models.csrnet import CSRNet
from preprocess import load_image


def predict_image(image_path, model_path="best_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = load_image(image_path)

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img = torch.tensor(img, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(img)

    density_map = output.squeeze().cpu().numpy()
    count = np.sum(density_map) / 1000

    print(f" Predicted Count: {int(count)}")

    return density_map, count