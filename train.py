import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.csrnet import CSRNet
from crowd_dataset import CrowdDataset


def main():
 
    # Device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
 
    # Dataset Paths
    
    image_path = "data/part_A_final/train_data/images"
    gt_path = "data/part_A_final/train_data/ground_truth"

    dataset = CrowdDataset(image_path, gt_path)

    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print("Total dataset:", total_size)
    print("Train size:", train_size)
    print("Validation size:", val_size)
    print("Test size:", test_size)

    
    # Split Dataset
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
 
    # Model
    
    model = CSRNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
 
    # Training Config
     
    num_epochs = 50

    train_losses = []
    val_losses = []

    print("Starting Training...")
 
    # Training Loop
     
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0

        for img, gt, _ in train_loader:

            img = img.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            output = model(img)

            if output.shape != gt.shape:
                gt = F.interpolate(
                    gt,
                    size=(output.shape[2], output.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(output, gt)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

         
        # Validation
         
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for img, gt, _ in val_loader:

                img = img.to(device)
                gt = gt.to(device)

                output = model(img)

                if output.shape != gt.shape:
                    gt = F.interpolate(
                        gt,
                        size=(output.shape[2], output.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = criterion(output, gt)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/50]  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
        )

    print("Training Finished")

     
    # Save Model
    
    torch.save(model.state_dict(), "csrnet_trained.pth")
    print("Model saved as csrnet_trained.pth")

     
    # Test Evaluation
     
    model.eval()
    test_loss = 0

    with torch.no_grad():

        for img, gt, _ in test_loader:

            img = img.to(device)
            gt = gt.to(device)

            output = model(img)

            if output.shape != gt.shape:
                gt = F.interpolate(
                    gt,
                    size=(output.shape[2], output.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(output, gt)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print("Test Loss:", test_loss)

     
    # Plot Loss Graph
     
    plt.figure()

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Loss")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()