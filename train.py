import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import CrowdDataset
from model import CSRNet


# 🔥 RENAMED FUNCTION
def train_model():

    # -----------------------------
    # Dataset Load (Train + Val)
    # -----------------------------
    dataset_paths = [
        "Dataset/ShanghaiTech/part_A/train_data",
        "Dataset/ShanghaiTech/part_B/train_data"
    ]

    train_dataset = CrowdDataset(dataset_paths, split="train")
    val_dataset   = CrowdDataset(dataset_paths, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -----------------------------
    # Device Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)

    # -----------------------------
    # Model
    # -----------------------------
    model = CSRNet().to(device)

    for param in model.frontend.parameters():
        param.requires_grad = False

    # -----------------------------
    # Loss and Optimizer
    # -----------------------------
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
    )

    # -----------------------------
    # AMP
    # -----------------------------
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -----------------------------
    # Training Settings
    # -----------------------------
    epochs = 50

    train_losses = []
    val_losses = []

    print("Training Started...")

    for epoch in range(epochs):

        if epoch == 10:
            print("Unfreezing frontend layers...")
            for param in model.frontend.parameters():
                param.requires_grad = True

        model.train()
        epoch_loss = 0

        for images, densities in train_loader:

            images = images.to(device)
            densities = densities.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, densities)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, densities)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, densities in val_loader:

                images = images.to(device)
                densities = densities.to(device)

                outputs = model(images)
                loss = criterion(outputs, densities)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "csrnet_model.pth")
    print("Model saved as csrnet_model.pth")

    # Plot
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()


# 🔥 IMPORTANT
if __name__ == "__main__":
    train_model()