import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from preprocess import load_image, load_annotation, generate_density_map
import config


class CrowdDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.image_dir = image_dir
        self.gt_dir = gt_dir

        self.image_paths = sorted(os.listdir(image_dir))
        self.gt_paths = sorted(os.listdir(gt_dir))

        assert len(self.image_paths) == len(self.gt_paths), \
            "❌ Images and GT count mismatch!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_name = self.image_paths[idx]
        gt_name = self.gt_paths[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mat_path = os.path.join(self.gt_dir, gt_name)

        # ✅ Already normalized in preprocess
        image = load_image(img_path)

        points = load_annotation(mat_path)

        # 🔥 Generate density
        density = generate_density_map(image.shape, points)

        # 🔥 Store original count BEFORE resize
        original_count = density.sum()

        # Resize
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        density = cv2.resize(
            density,
            (config.DENSITY_SIZE, config.DENSITY_SIZE)
        )

        # 🔥 CRITICAL FIX → preserve total count
        if density.sum() > 0:
            density = density * (original_count / density.sum())

        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1)
        density = torch.tensor(density).unsqueeze(0)

        return image.float(), density.float()


def get_dataloaders():
    dataset = CrowdDataset(config.IMAGE_DIR, config.GT_DIR)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # ✅ safe for Windows
        pin_memory=False
    )
