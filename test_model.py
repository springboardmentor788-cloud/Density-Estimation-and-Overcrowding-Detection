import torch
from torch.utils.data import DataLoader
from dataset import CrowdDataset
from model import CSRNet
import os


def test_model():

    # ------------------------
    # Device Check
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using Device:", device)

    # ------------------------
    # Load Test Dataset
    # ------------------------
    dataset_paths = [
        "C:/Users/tarun/deepvision/Dataset/ShanghaiTech/part_A/train_data",
        "C:/Users/tarun/deepvision/Dataset/ShanghaiTech/part_B/train_data"
    ]

    # ✅ Check each dataset path
    print("📂 Checking Dataset Paths...")
    for path in dataset_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Path not found: {path}")
        print("✔", path)

    # ✅ Load dataset correctly
    test_dataset = CrowdDataset(dataset_paths, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"✅ Total Test Samples: {len(test_dataset)}")

    # ------------------------
    # Load Model
    # ------------------------
    model_path = "csrnet_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = CSRNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Model Loaded Successfully")

    # ------------------------
    # Evaluation (MAE)
    # ------------------------
    total_error = 0

    with torch.no_grad():
        for i, (img, gt) in enumerate(test_loader):

            img = img.to(device)
            gt = gt.to(device)

            output = model(img)

            pred_count = output.sum().item()
            gt_count = gt.sum().item()

            error = abs(pred_count - gt_count)
            total_error += error

            # 🔥 Per image debug
            print(f"[{i+1}] Pred: {pred_count:.2f} | GT: {gt_count:.2f} | Error: {error:.2f}")

    mae = total_error / len(test_loader)

    print("\n🎯 Final Test MAE:", mae)


if __name__ == "__main__":
    test_model()