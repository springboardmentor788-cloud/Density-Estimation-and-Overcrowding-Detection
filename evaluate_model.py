import torch
from torch.utils.data import DataLoader
from dataset_loader import ShanghaiTechDataset
from models.csrnet import CSRNet

test_img_path = r"dataset/ShanghaiTech/part_A/test_data/images"
test_gt_path = r"dataset/ShanghaiTech/part_A/test_data/ground-truth"

dataset = ShanghaiTechDataset(test_img_path, test_gt_path)
loader = DataLoader(dataset, batch_size=1)

model = CSRNet()
model.load_state_dict(torch.load("crowd_model.pth"))
model.eval()

mae = 0
mse = 0
count = 0

with torch.no_grad():

    for img, density in loader:

        pred = model(img)

        pred_count = pred.sum().item()
        gt_count = density.sum().item()

        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count) ** 2

        count += 1

mae = mae / count
rmse = (mse / count) ** 0.5

print("MAE:", mae)
print("RMSE:", rmse)