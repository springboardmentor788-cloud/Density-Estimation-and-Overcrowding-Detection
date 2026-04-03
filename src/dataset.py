from pathlib import Path
from typing import Tuple, Union, Optional

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_ROOT, PREPROCESSED_ROOT, PreprocessConfig
from density_utils import points_to_density_map_simple


def load_shanghaitech_points(mat_path: Path) -> np.ndarray:
    """Load head coordinates from ShanghaiTech .mat file. Returns (N, 2) array in (x, y)."""
    data = scipy.io.loadmat(str(mat_path))
    if "image_info" in data:
        info = data["image_info"]
        try:
            arr = info[0][0][0][0][0]
        except (IndexError, TypeError):
            try:
                arr = info[0][0][0][0]
            except (IndexError, TypeError):
                arr = info[0][0]
        pts = np.asarray(arr)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        elif pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T
        return np.asarray(pts).reshape(-1, 2).astype(np.float64)
    for key in ("annPoints", "points", "location"):
        if key in data:
            return np.asarray(data[key]).reshape(-1, 2).astype(np.float64)
    raise KeyError(f"No point data found in {mat_path}")


class ShanghaiTechImageDataset(Dataset):
    """
    Minimal dataset for loading preprocessed ShanghaiTech images.

    This is sufficient for Milestone 1:
    - image resizing is already done in preprocess.py
    - normalization (mean/std) is applied here before feeding to models
    """

    def __init__(
        self,
        root: Union[Path, str, None] = None,
        part: str = "A",
        split: str = "train",
        transform=None,
    ):
        """
        Args:
            root: Preprocessed dataset root. Defaults to PREPROCESSED_ROOT.
            part: "A" or "B"
            split: "train" or "test"
            transform: Optional torchvision-style transform.
        """
        if root is None:
            root = PREPROCESSED_ROOT
        self.root = Path(root)
        self.part = part
        self.split = split

        self.img_dir = self.root / f"part_{part}" / split / "images"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                "Run `python preprocess.py` after placing the ShanghaiTech dataset."
            )

        self.image_paths = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in {self.img_dir}. "
                "Check that preprocessing completed successfully."
            )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # HWC [0,255] -> CHW [0,1]
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img)
        return img_tensor, str(img_path)


class ShanghaiTechDensityDataset(Dataset):
    """
    Dataset for training density models: loads raw image + .mat, resizes image,
    generates ground-truth density map at 1/8 resolution. Returns (image, density_map, count).
    """

    def __init__(
        self,
        root: Union[Path, str, None] = None,
        part: str = "A",
        split: str = "train",
        transform=None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        if root is None:
            root = DATA_ROOT
        self.root = Path(root)
        self.part = part
        self.split = split
        self.target_size = target_size or PreprocessConfig.TARGET_SIZE  # (W, H)
        self.density_h = self.target_size[1] // 8
        self.density_w = self.target_size[0] // 8

        self.img_dir = self.root / f"part_{part}_final" / f"{split}_data" / "images"
        # Support both "ground_truth" and "ground-truth" folder names
        base_gt = self.root / f"part_{part}_final" / f"{split}_data"
        gt_candidate = base_gt / "ground_truth"
        gt_candidate_hyphen = base_gt / "ground-truth"
        self.gt_dir = gt_candidate if gt_candidate.exists() else gt_candidate_hyphen

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                "Place the ShanghaiTech dataset under data/ShanghaiTech."
            )

        self.image_paths = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images in {self.img_dir}")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.image_paths) * 2
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        is_augmented = False
        if self.split == "train" and idx >= len(self.image_paths):
            idx = idx - len(self.image_paths)
            is_augmented = True

        img_path = self.image_paths[idx]
        # GT_IMG_1.mat for IMG_1.jpg
        stem = img_path.stem
        if stem.startswith("IMG_"):
            gt_name = f"GT_{stem}.mat"
        else:
            gt_name = f"GT_{stem}.mat"
        gt_path = self.gt_dir / gt_name

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Resize image to target
        img = cv2.resize(
            img,
            (self.target_size[0], self.target_size[1]),
            interpolation=cv2.INTER_CUBIC,
        )

        if is_augmented:
            img = cv2.flip(img, 1)

        img_tensor = self.transform(img)

        # Load points and scale to density map resolution (1/8)
        if gt_path.exists():
            points = load_shanghaitech_points(gt_path)
            # Scale from original image coords to density map coords
            scale_x = self.density_w / orig_w
            scale_y = self.density_h / orig_h
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            density = points_to_density_map_simple(
                points, self.density_h, self.density_w, kernel_size=15, sigma=2.0
            )
            count = float(density.sum())
        else:
            density = np.zeros((self.density_h, self.density_w), dtype=np.float32)
            count = 0.0

        if is_augmented:
            density = np.fliplr(density).copy()

        density_tensor = torch.from_numpy(density).unsqueeze(0)  # (1, H, W)
        return img_tensor, density_tensor, count


__all__ = [
    "ShanghaiTechImageDataset",
    "ShanghaiTechDensityDataset",
    "load_shanghaitech_points",
]

