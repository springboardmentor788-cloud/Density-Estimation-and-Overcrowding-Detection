import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models.csrnet import CSRNet
import config


def train():

    print("🚀 Training started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSRNet().to(device)

    # ✅ LOAD PREVIOUS MODEL IF EXISTS
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("✅ Loaded existing best_model.pth (Resuming training)")
    else:
        print("⚠️ No pretrained model found, training from scratch")

    print("✅ Training full model (frontend + backend)")

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    loader = get_dataloaders()

    losses = []
    maes = []

    best_mae = float("inf")  # 🔥 Track best model

    for epoch in range(config.EPOCHS):

        model.train()

        epoch_loss = 0
        epoch_mae = 0

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ✅ MAE calculation
            pred_count = outputs.detach().cpu().numpy().sum()
            gt_count = targets.detach().cpu().numpy().sum()

            epoch_mae += abs(pred_count - gt_count)

        avg_loss = epoch_loss / len(loader)
        avg_mae = epoch_mae / len(loader)

        losses.append(avg_loss)
        maes.append(avg_mae)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {avg_loss:.4f} | MAE: {avg_mae:.2f}")

        # ✅ SAVE ONLY BEST MODEL
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 Saved NEW best model (MAE: {best_mae:.2f})")

    print("✅ Training complete")

    # 📊 Plot Loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")

    # 📊 Plot MAE
    plt.figure()
    plt.plot(maes)
    plt.title("MAE (Count Error)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.savefig("mae_plot.png")

    print("📊 Plots saved")


if __name__ == "__main__":
    train()
