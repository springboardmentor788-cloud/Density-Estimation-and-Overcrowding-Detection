import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from config import PROJECT_ROOT
from csrnet import CSRNet, density_to_count
from dataset import ShanghaiTechDensityDataset


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_stage(model, train_loader, val_loader, optimizer, criterion, device, epochs, stage_name, save_dir):
    print(f"\n--- Starting {stage_name} ---")
    
    early_stopping = EarlyStopping(patience=3, verbose=True)
    best_model_path = save_dir / f"best_model_{stage_name}.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for imgs, densities, counts in tqdm(train_loader, desc=f"{stage_name} - Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device)
            densities = densities.to(device)
            
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, densities)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        val_mae = float('inf')
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
            val_mae = mae_sum / n_val if n_val > 0 else float('inf')
            
        print(f"{stage_name} Epoch {epoch}: Train Loss = {train_loss:.6f}, Val MAE = {val_mae:.2f}")
        
        # Early Stopping check using Validation MAE
        early_stopping(val_mae, model, best_model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    # Load the best weights from this stage
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Incremental Training for CSRNet")
    parser.add_argument("--part", type=str, default="A")
    parser.add_argument("--batch_size", type=int, default=2, help="Lower batch size for laptop training")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--replay_ratio", type=float, default=0.1, help="Fraction of previous data to keep to avoid catastrophic forgetting")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = PROJECT_ROOT / "checkpoints" / "incremental_training"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Full Datasets
    full_train_set = ShanghaiTechDensityDataset(part=args.part, split="train")
    try:
        val_set = ShanghaiTechDensityDataset(part=args.part, split="test")
    except Exception:
        val_set = None

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False) if val_set else None

    # Determine chunks safely (to handle exact 600 or dynamically)
    total_images = len(full_train_set)
    chunk_size = total_images // 3
    
    print(f"Total training images (including augmentations): {total_images}")
    print(f"Chunk size ~ {chunk_size} images per stage")

    # Indices for the 3 parts
    indices = list(range(total_images))
    random.shuffle(indices) # Shuffle all indices to ensure varied distribution
    
    part1_idx = indices[:chunk_size]
    part2_idx = indices[chunk_size:2*chunk_size]
    part3_idx = indices[2*chunk_size:]

    # 2. Setup Model & Optimizer
    model = CSRNet(pretrained=True).to(device)
    criterion = nn.MSELoss(reduction="mean")

    # ---------------- Stage 1 ----------------
    # First part images -> Train for 12 epochs
    stage1_dataset = Subset(full_train_set, part1_idx)
    loader1 = DataLoader(stage1_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model = train_stage(model, loader1, val_loader, optimizer1, criterion, device, epochs=12, stage_name="Stage_1", save_dir=save_dir)

    # ---------------- Stage 2 ----------------
    # Second part images + Replay buffer from Part 1 -> Train for 8 epochs
    replay_size_1 = int(len(part1_idx) * args.replay_ratio)
    replay_idx_1 = random.sample(part1_idx, replay_size_1)
    
    stage2_dataset = Subset(full_train_set, part2_idx + replay_idx_1)
    loader2 = DataLoader(stage2_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Lower learning rate for continued training to prevent destroying old weights
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5)
    model = train_stage(model, loader2, val_loader, optimizer2, criterion, device, epochs=8, stage_name="Stage_2", save_dir=save_dir)

    # ---------------- Stage 3 ----------------
    # Third part images + Replay buffer from Parts 1 & 2 -> Train for 8 epochs
    replay_size_2 = int((len(part1_idx) + len(part2_idx)) * args.replay_ratio)
    replay_idx_2 = random.sample(part1_idx + part2_idx, replay_size_2)
    
    stage3_dataset = Subset(full_train_set, part3_idx + replay_idx_2)
    loader3 = DataLoader(stage3_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    optimizer3 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.25)
    model = train_stage(model, loader3, val_loader, optimizer3, criterion, device, epochs=8, stage_name="Stage_3", save_dir=save_dir)

    print("Incremental training completed successfully!")

if __name__ == "__main__":
    main()
