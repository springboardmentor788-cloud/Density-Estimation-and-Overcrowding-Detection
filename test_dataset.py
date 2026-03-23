from torch.utils.data import DataLoader
from dataset_loader import ShanghaiTechDataset

img_path = r"C:/Users/Online/Desktop/DeepVision/dataset/ShanghaiTech/part_A/train_data/images"
gt_path = r"C:/Users/Online/Desktop/DeepVision/dataset/ShanghaiTech/part_A/train_data/ground-truth"

dataset = ShanghaiTechDataset(img_path, gt_path)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for img, density in loader:
    print("Image batch:", img.shape)
    print("Density batch:", density.shape)
    break