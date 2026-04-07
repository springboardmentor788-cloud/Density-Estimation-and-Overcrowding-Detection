from torch.utils.data import DataLoader, random_split
import torch

from dataset import CrowdDataset
from visualize import show_image, show_density_map
from models.csrnet import CSRNet
from train import train
import config


# ----------------------------
# Milestone 1 : Dataset Test
# ----------------------------
def run_milestone1():

    print("Running Milestone 1 verification")

    dataset = CrowdDataset(config.IMAGE_DIR, config.GT_DIR)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0   # important for Windows
    )

    for images, densities in loader:

        print("Image tensor shape:", images.shape)
        print("Density map shape:", densities.shape)

        show_image(images[0])
        show_density_map(densities[0])
        break


# ----------------------------
# Milestone 2 : Training
# ----------------------------
def run_training():

    print("Starting training pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = CrowdDataset(config.IMAGE_DIR, config.GT_DIR)

    # 🔹 Train / Validation Split (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 🔹 Model
    model = CSRNet()

    # 🔹 Start Training (20 epochs as suggested)
    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=config.EPOCHS
    )


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    MODE = "train"   # change to "test" for milestone 1

    if MODE == "test":
        run_milestone1()

    elif MODE == "train":
        run_training()