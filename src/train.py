"""
Milestone 2: Training script for CSRNet on ShanghaiTech.
Trains on a subset, validates, reports MAE and loss; saves best model.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PROJECT_ROOT
from csrnet import CSRNet, density_to_count
from dataset import ShanghaiTechDensityDataset


def main():
    parser = argparse.ArgumentParser(description="Train CSRNet on ShanghaiTech")
    parser.add_argument("--part", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--subset", type=int, default=0, help="Use only N train samples (0 = all)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import datetime
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_subset{args.subset}_epochs{args.epochs}_{timestamp}"
    save_dir = Path(args.save_dir or str(PROJECT_ROOT / "checkpoints" / run_name))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {save_dir}")

    # Data
    train_set = ShanghaiTechDensityDataset(part=args.part, split="train")
    if args.subset > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(args.subset, len(train_set))))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    try:
        val_set = ShanghaiTechDensityDataset(part=args.part, split="test")
    except Exception:
        val_set = None
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ) if val_set else None

    # Model and optimizer
    model = CSRNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss(reduction="mean")

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, densities, counts in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device)
            densities = densities.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, densities)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation: MAE on count
        if val_loader:
            model.eval()
            mae_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for imgs, _, counts in val_loader:
                    imgs = imgs.to(device)
                    pred = model(imgs)
                    pred_counts = density_to_count(pred).cpu().numpy()
                    for i in range(imgs.size(0)):
                        c = counts[i].item() if torch.is_tensor(counts[i]) else counts[i]
                        mae_sum += abs(float(pred_counts[i]) - float(c))
                        n_val += 1
            val_mae = mae_sum / n_val if n_val else float("inf")
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "mae": val_mae},
                    save_dir / "csrnet_best.pt",
                )
            print(f"Epoch {epoch}  train_loss={train_loss:.6f}  val_mae={val_mae:.2f}  best_mae={best_mae:.2f}")
        else:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                save_dir / "csrnet_last.pt",
            )
            print(f"Epoch {epoch}  train_loss={train_loss:.6f}")

    print(f"Training done. Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()
