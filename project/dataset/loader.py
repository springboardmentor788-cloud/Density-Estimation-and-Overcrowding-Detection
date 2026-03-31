from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from project.config import IMAGENET_MEAN, IMAGENET_STD
from project.dataset.density_map import load_or_create_density_map


def get_part_root(dataset_root: Path, part: str) -> Path:
    part = part.upper()
    if part not in {"A", "B"}:
        raise ValueError("part must be one of: A, B")
    return dataset_root / f"part_{part}_final"


def list_image_paths(dataset_root: Path, part: str, split: str) -> List[Path]:
    part_root = get_part_root(dataset_root, part)
    image_dir = part_root / f"{split}_data" / "images"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def image_to_mat_path(image_path: Path) -> Path:
    gt_dir = image_path.parent.parent / "ground_truth"
    mat_name = image_path.stem.replace("IMG_", "GT_IMG_") + ".mat"
    return gt_dir / mat_name


def resize_with_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale == 1.0:
        return image

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class CrowdCountingDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        part: str,
        split: str,
        cache_root: Path,
        max_dim: int = 1536,
        crop_size: int = 512,
        output_stride: int = 8,
        train: bool = True,
        random_flip: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.part = part
        self.split = split
        self.cache_root = Path(cache_root)
        self.max_dim = max_dim
        self.crop_size = crop_size
        self.output_stride = output_stride
        self.train = train
        self.random_flip = random_flip

        self.image_paths = list_image_paths(self.dataset_root, part, split)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float()

    def _crop_pair(self, image: np.ndarray, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.train:
            # For evaluation keep full frame but enforce output-stride-aligned spatial shape.
            ih, iw = image.shape[:2]
            ih_aligned = max(self.output_stride, (ih // self.output_stride) * self.output_stride)
            iw_aligned = max(self.output_stride, (iw // self.output_stride) * self.output_stride)
            image = image[:ih_aligned, :iw_aligned]
            d_h = ih_aligned // self.output_stride
            d_w = iw_aligned // self.output_stride
            density = density[:d_h, :d_w]
            return np.ascontiguousarray(image), np.ascontiguousarray(density)

        ih, iw = image.shape[:2]
        ch = min(self.crop_size, ih)
        cw = min(self.crop_size, iw)

        if ih == ch:
            top = 0
        else:
            top = random.randint(0, ih - ch)

        if iw == cw:
            left = 0
        else:
            left = random.randint(0, iw - cw)

        image_crop = image[top : top + ch, left : left + cw]

        d_top = top // self.output_stride
        d_left = left // self.output_stride
        d_ch = max(1, ch // self.output_stride)
        d_cw = max(1, cw // self.output_stride)
        density_crop = density[d_top : d_top + d_ch, d_left : d_left + d_cw]

        # Force fixed-size crops for batch collation stability.
        tgt_h = self.crop_size
        tgt_w = self.crop_size
        if image_crop.shape[0] != tgt_h or image_crop.shape[1] != tgt_w:
            image_crop = cv2.resize(image_crop, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

        tgt_dh = max(1, tgt_h // self.output_stride)
        tgt_dw = max(1, tgt_w // self.output_stride)
        if density_crop.shape[0] != tgt_dh or density_crop.shape[1] != tgt_dw:
            density_crop = cv2.resize(density_crop, (tgt_dw, tgt_dh), interpolation=cv2.INTER_AREA)
            orig_sum = float(density[d_top : d_top + d_ch, d_left : d_left + d_cw].sum())
            new_sum = float(density_crop.sum())
            if new_sum > 0:
                density_crop *= orig_sum / new_sum

        return np.ascontiguousarray(image_crop), np.ascontiguousarray(density_crop)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mat_path = image_to_mat_path(image_path)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        image_rgb = resize_with_max_dim(image_rgb, self.max_dim)

        h, w = image_rgb.shape[:2]
        scale_x = float(w) / float(max(orig_w, 1))
        scale_y = float(h) / float(max(orig_h, 1))
        density = load_or_create_density_map(
            image_path=image_path,
            mat_path=mat_path,
            image_shape=(h, w),
            cache_root=self.cache_root / f"part_{self.part.upper()}_{self.split}",
            output_stride=self.output_stride,
            point_scale=(scale_x, scale_y),
        )

        image_rgb, density = self._crop_pair(image_rgb, density)

        if self.train and self.random_flip and random.random() < 0.5:
            image_rgb = np.ascontiguousarray(np.fliplr(image_rgb))
            density = np.ascontiguousarray(np.fliplr(density))

        image_tensor = self._normalize(image_rgb)
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        if not self.train:
            # Pad eval samples to a common size per batch-friendly stride multiples.
            h, w = image_tensor.shape[1:]
            d_h, d_w = density_tensor.shape[1:]
            h_aligned = max(self.output_stride, ((h + self.output_stride - 1) // self.output_stride) * self.output_stride)
            w_aligned = max(self.output_stride, ((w + self.output_stride - 1) // self.output_stride) * self.output_stride)
            if h != h_aligned or w != w_aligned:
                image_tensor = F.pad(image_tensor, (0, w_aligned - w, 0, h_aligned - h), mode="constant", value=0.0)
            target_dh = h_aligned // self.output_stride
            target_dw = w_aligned // self.output_stride
            if d_h != target_dh or d_w != target_dw:
                density_tensor = F.pad(density_tensor, (0, target_dw - d_w, 0, target_dh - d_h), mode="constant", value=0.0)

        count = density_tensor.sum().item()

        return {
            "image": image_tensor,
            "density": density_tensor,
            "count": torch.tensor(count, dtype=torch.float32),
            "image_path": str(image_path),
        }


def create_dataloader(
    dataset_root: Path,
    part: str,
    split: str,
    cache_root: Path,
    batch_size: int,
    workers: int,
    max_dim: int,
    crop_size: int,
    output_stride: int,
    train: bool,
) -> DataLoader:
    dataset = CrowdCountingDataset(
        dataset_root=dataset_root,
        part=part,
        split=split,
        cache_root=cache_root,
        max_dim=max_dim,
        crop_size=crop_size,
        output_stride=output_stride,
        train=train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )
