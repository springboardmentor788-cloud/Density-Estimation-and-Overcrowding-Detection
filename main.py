import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
 
    # Resize Image to 224x224
     
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    # Print Input Shape
    print("Input Image Shape:", img.shape)
 
    # Model Prediction
     
    with torch.no_grad():
        output = model(img)

    # Print original density map shape
    print("Original Density Map Shape:", output.shape)
 
    # Resize density map to 224x224
 
    output_resized = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

    # Print resized density map shape
    print("Resized Density Map Shape:", output_resized.shape)
 
    # Predicted Count
    
    predicted_count = output.sum().item()
    print("Predicted Count:", round(predicted_count, 2))
 
    # Show Density Map
     
    density_map = output_resized.squeeze().cpu().numpy()

    plt.imshow(density_map, cmap="jet")
    plt.title("Density Map (224x224)")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()