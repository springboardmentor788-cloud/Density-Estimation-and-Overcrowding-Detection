import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io as sio


class CrowdDataset(Dataset):

    def __init__(self, img_path, gt_path):

        self.img_path = img_path
        self.gt_path = gt_path

        self.image_list = [f for f in os.listdir(img_path) if f.endswith(".jpg")]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_name = self.image_list[idx]

        img_file = os.path.join(self.img_path, img_name)
        gt_file = os.path.join(self.gt_path, "GT_" + img_name.replace(".jpg",".mat"))

        # Load image
        img = Image.open(img_file).convert("RGB")
        width, height = img.size

        img = self.transform(img)

        # Load ground truth
        mat = sio.loadmat(gt_file)
        points = mat["image_info"][0][0][0][0][0]

        density = torch.zeros((1,224,224))

        # scale factors
        scale_x = 224 / width
        scale_y = 224 / height

        for p in points:

            x = int(p[0] * scale_x)
            y = int(p[1] * scale_y)

            if x >= 224:
                x = 223
            if y >= 224:
                y = 223

            density[0,y,x] = 1

        count = len(points)

        return img, density, count