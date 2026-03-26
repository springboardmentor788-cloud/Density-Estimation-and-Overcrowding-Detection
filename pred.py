import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.csrnet import CSRNet


def main():

    
    # Device
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

     
    # Load Model
     
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")
 
    # Image Path
     
    image_path = "data/part_A_final/test_data/images/IMG_1.jpg"

     
    # Transform (Resize 224x224) 
     
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load image
    img = Image.open(image_path).convert("RGB")
    img = transform(img)

    # Add batch dimension
    img = img.unsqueeze(0).to(device)

     
    # Print Image Tensor Shape
     
    print("Input Image Tensor Shape:", img.shape)

     
    # Prediction
     
    with torch.no_grad():
        output = model(img)

     
    # Print Density Map Shape
     
    print("Density Map Tensor Shape:", output.shape)

    # Predicted Count
    predicted_count = output.sum().item()
    print("Predicted Count:", round(predicted_count, 2))

     
    # Show Density Map
    
    density_map = output.squeeze().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(density_map, cmap="jet")
    plt.colorbar()
    plt.title("Predicted Density Map")
    plt.show()


if __name__ == "__main__":
    main()