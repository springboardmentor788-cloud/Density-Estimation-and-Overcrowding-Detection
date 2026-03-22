import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from scripts.dataset_loader import ShanghaiTechDataset

# Dataset path
DATASET_PATH = "data/ShanghaiTech/part_A/train_data/images"

# Resize settings
RESIZE_HEIGHT = 480
RESIZE_WIDTH = 640


def main():

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = ShanghaiTechDataset(DATASET_PATH, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Total Images in Dataset: {len(dataset)}")

    # Check one batch
    for batch in dataloader:
        print(f"Batch Shape: {batch.shape}")
        break


if __name__ == "__main__":
    main()