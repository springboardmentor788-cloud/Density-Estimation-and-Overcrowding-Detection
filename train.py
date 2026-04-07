import matplotlib
matplotlib.use('TkAgg')  # Fix for Windows (graph display)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ================= TRAIN FUNCTION =================
def train(model, train_loader, val_loader, device, epochs=20, lr=1e-5):

    print("🚀 Starting training...")

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_mae = float("inf")

    train_losses = []
    val_mae_list = []

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        for batch_idx, (images, density_maps) in enumerate(train_loader):

            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)

            # Debug shapes (first batch only)
            if epoch == 0 and batch_idx == 0:
                print("Image shape:", images.shape)
                print("GT shape:", density_maps.shape)
                print("Output shape:", outputs.shape)

            loss = criterion(outputs, density_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"\n📌 Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {avg_loss:.4f}")

        # Validation
        val_mae = validate(model, val_loader, device)
        val_mae_list.append(val_mae)

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

        # Save every epoch
        torch.save(model.state_dict(), f"csrnet_epoch_{epoch+1}.pth")

    # Plot graphs
    print("📊 Plotting graphs...")
    plot_training_curves(train_losses, val_mae_list)


# ================= VALIDATION =================
def validate(model, val_loader, device):

    model.eval()
    mae = 0

    with torch.no_grad():
        for images, density_maps in val_loader:

            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)

            pred_count = outputs.sum().item()
            gt_count = density_maps.sum().item()

            mae += abs(pred_count - gt_count)

    mae /= len(val_loader)

    print(f"Validation MAE: {mae:.2f}")
    return mae


# ================= GRAPHING =================
def plot_training_curves(train_losses, val_mae_list):

    # Loss curve
    plt.figure()
    plt.plot(train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.savefig("loss_curve.png")
    plt.close()

    # MAE curve
    plt.figure()
    plt.plot(val_mae_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("Validation MAE Curve")
    plt.grid()
    plt.savefig("val_mae_curve.png")
    plt.close()

    print("✅ Graphs saved: loss_curve.png, val_mae_curve.png")


# ================= MAIN EXECUTION =================
if __name__ == "__main__":

    print("🚀 train.py is running...")

    # 🔹 IMPORT YOUR FILES
    from dataset import get_dataloaders
    from models.csrnet import CSRNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 🔹 LOAD DATA
    train_loader, val_loader = get_dataloaders()
    print("✅ Data loaded")

    # 🔹 CREATE MODEL
    model = CSRNet()
    print("✅ Model created")

    # 🔹 START TRAINING
    train(model, train_loader, val_loader, device)