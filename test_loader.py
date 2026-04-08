from data_loader import CrowdDataset

dataset = CrowdDataset(
    "dataset/ShanghaiTech/part_A/train_data/images",
    "dataset/ShanghaiTech/part_A/train_data/ground-truth"
)

img, gt = dataset[0]

print("Image shape:", img.shape)
print("Density shape:", gt.shape)