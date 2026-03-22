import cv2
import torch
import scipy.io as sio
from torch.utils.data import Dataset

from density_utils import generate_density_map


class CrowdDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Load ground truth
        mat_path = img_path.replace("images", "ground-truth")
        mat_path = mat_path.replace(".jpg", ".mat")
        mat_path = mat_path.replace("IMG_", "GT_IMG_")

        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]

        # Generate density map
        density = generate_density_map((h, w), points)

        # Resize image
        img = cv2.resize(img, (512, 512))
        density = cv2.resize(density, (64, 64))

        # 🔥 CRITICAL FIX (DO NOT REMOVE)
        density = density * (h * w) / (28 * 28)

        # Convert to tensor
        img = torch.tensor(img).permute(2, 0, 1).float() / 255
        density = torch.tensor(density).unsqueeze(0).float()

        count = len(points)

        # Debug (REMOVE after checking)
        

        return img, density, count
   