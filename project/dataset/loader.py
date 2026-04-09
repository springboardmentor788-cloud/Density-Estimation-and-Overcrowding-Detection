from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import torch
    from torch.utils.data import ConcatDataset, Dataset
except Exception:  # pragma: no cover
    torch = None
    ConcatDataset = object
    Dataset = object

from dataset.density_map import (
    density_cache_path,
    load_or_create_density_map,
    load_points_from_mat_cached,
)
from config import CONFIG


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_image(path: str | Path) -> np.ndarray:
    path = str(path)
    if cv2 is not None:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if Image is None:
        raise RuntimeError("Neither cv2 nor PIL is available for image loading")
    return np.asarray(Image.open(path).convert("RGB"))


def _resize_image(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if Image is None:
        raise RuntimeError("Neither cv2 nor PIL is available for image resizing")
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def _resize_density_map(density: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if density.shape[:2] == (height, width):
        return density.astype(np.float32)
    if cv2 is not None:
        resized = cv2.resize(density.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    else:
        if Image is None:
            raise RuntimeError("Neither cv2 nor PIL is available for density resizing")
        resized = np.asarray(Image.fromarray(density.astype(np.float32)).resize((width, height), Image.BILINEAR), dtype=np.float32)
    scale = (density.shape[0] * density.shape[1]) / float(height * width)
    return resized.astype(np.float32) * scale


def _flip_points_horizontal(points: np.ndarray, width: int) -> np.ndarray:
    flipped = points.copy()
    flipped[:, 0] = (width - 1) - flipped[:, 0]
    return flipped


def _crop_image_and_points(
    image: np.ndarray,
    points: np.ndarray,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    cropped = image[top : top + crop_h, left : left + crop_w]
    shifted = points.copy()
    shifted[:, 0] -= left
    shifted[:, 1] -= top
    mask = (
        (shifted[:, 0] >= 0)
        & (shifted[:, 0] < crop_w)
        & (shifted[:, 1] >= 0)
        & (shifted[:, 1] < crop_h)
    )
    return cropped, shifted[mask]


@dataclass(slots=True)
class CrowdSample:
    sample_id: str
    image_path: Path
    mat_path: Path


class CrowdCountDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.2,
        resize_to: tuple[int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
        random_flip: bool = True,
        random_crop: bool = False,
        use_cache: bool = True,
        sigma_scale: float = 0.3,
        min_sigma: float = 4.0,
        knn: int = 4,
        seed: int = 42,
        max_samples: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.val_fraction = float(val_fraction)
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.use_cache = use_cache
        self.sigma_scale = sigma_scale
        self.min_sigma = min_sigma
        self.knn = knn
        self.seed = seed
        self.max_samples = max_samples

        if self.split not in {"train", "val", "test", "predict"}:
            raise ValueError(f"Unsupported split: {split}")

        if self.split in {"test", "predict"}:
            image_dir = self.data_root / "test_data" / "images"
            gt_dir = self.data_root / "test_data" / "ground_truth"
            self.samples = self._collect_samples(image_dir, gt_dir)
        else:
            image_dir = self.data_root / "train_data" / "images"
            gt_dir = self.data_root / "train_data" / "ground_truth"
            samples = self._collect_samples(image_dir, gt_dir)
            rng = random.Random(self.seed)
            rng.shuffle(samples)
            split_index = max(1, int(round(len(samples) * (1.0 - self.val_fraction))))
            if self.split == "train":
                self.samples = samples[:split_index]
            else:
                self.samples = samples[split_index:]

        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def _collect_samples(self, image_dir: Path, gt_dir: Path) -> list[CrowdSample]:
        samples: list[CrowdSample] = []
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            sample_id = image_path.stem
            mat_name = sample_id.replace("IMG", "GT_IMG") + ".mat"
            mat_path = gt_dir / mat_name
            if not mat_path.exists():
                raise FileNotFoundError(mat_path)
            samples.append(CrowdSample(sample_id=sample_id, image_path=image_path, mat_path=mat_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_augment(
        self, image: np.ndarray, points: np.ndarray, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]

        if self.random_flip and rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            points = _flip_points_horizontal(points, width)

        if self.random_crop and self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            if height >= crop_h and width >= crop_w:
                top = rng.randint(0, height - crop_h)
                left = rng.randint(0, width - crop_w)
                image, points = _crop_image_and_points(image, points, top, left, crop_h, crop_w)

        return image, points

    def _resize_points(self, points: np.ndarray, original_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
        original_h, original_w = original_shape
        target_h, target_w = target_shape
        if original_h == target_h and original_w == target_w:
            return points

        scale_x = target_w / float(original_w)
        scale_y = target_h / float(original_h)
        resized = points.copy().astype(np.float32)
        resized[:, 0] *= scale_x
        resized[:, 1] *= scale_y
        return resized

    def _augment_density(self, density: np.ndarray, image: np.ndarray, points: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]

        if self.random_flip and rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            density = np.ascontiguousarray(density[:, ::-1])
            points = _flip_points_horizontal(points, width)

        if self.random_crop and self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            if height >= crop_h and width >= crop_w:
                top = rng.randint(0, height - crop_h)
                left = rng.randint(0, width - crop_w)
                image, points = _crop_image_and_points(image, points, top, left, crop_h, crop_w)
                density = density[top : top + crop_h, left : left + crop_w]

        return image, density, points

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = _read_image(sample.image_path)
        points = load_points_from_mat_cached(str(sample.mat_path)).copy()
        base_density = load_or_create_density_map(
            image.shape[:2],
            points,
            density_cache_path(self.data_root / "cache" / "density_maps", sample.sample_id, image.shape[:2]) if self.use_cache else None,
            knn=self.knn,
            sigma_scale=self.sigma_scale,
            min_sigma=self.min_sigma,
        )

        rng = random.Random(self.seed + index)
        if self.split == "train":
            image, base_density, points = self._augment_density(base_density, image, points, rng)

        if self.resize_to is not None:
            original_shape = image.shape[:2]
            image = _resize_image(image, self.resize_to)
            points = self._resize_points(points, original_shape, self.resize_to)
            base_density = _resize_density_map(base_density, self.resize_to)

        height, width = image.shape[:2]
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(CONFIG.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(CONFIG.image_std, dtype=torch.float32).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        density_tensor = torch.from_numpy(base_density.astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "density": density_tensor,
            "count": float(base_density.sum()),
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mat_path": str(sample.mat_path),
            "shape": (height, width),
        }


def build_crowd_datasets(
    data_roots: Iterable[str | Path],
    *,
    val_fraction: float = 0.2,
    resize_to: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
    random_flip: bool = True,
    random_crop: bool = False,
    use_cache: bool = True,
    sigma_scale: float = 0.3,
    min_sigma: float = 4.0,
    knn: int = 4,
    seed: int = 42,
    max_samples: int | None = None,
):
    train_sets = []
    val_sets = []
    test_sets = []

    for data_root in data_roots:
        train_sets.append(
            CrowdCountDataset(
                data_root,
                split="train",
                val_fraction=val_fraction,
                resize_to=resize_to,
                crop_size=crop_size,
                random_flip=random_flip,
                random_crop=random_crop,
                use_cache=use_cache,
                sigma_scale=sigma_scale,
                min_sigma=min_sigma,
                knn=knn,
                seed=seed,
                max_samples=max_samples,
            )
        )
        val_sets.append(
            CrowdCountDataset(
                data_root,
                split="val",
                val_fraction=val_fraction,
                resize_to=resize_to,
                crop_size=None,
                random_flip=False,
                random_crop=False,
                use_cache=use_cache,
                sigma_scale=sigma_scale,
                min_sigma=min_sigma,
                knn=knn,
                seed=seed,
                max_samples=max_samples,
            )
        )
        test_sets.append(
            CrowdCountDataset(
                data_root,
                split="test",
                resize_to=resize_to,
                crop_size=None,
                random_flip=False,
                random_crop=False,
                use_cache=use_cache,
                sigma_scale=sigma_scale,
                min_sigma=min_sigma,
                knn=knn,
                seed=seed,
                max_samples=max_samples,
            )
        )

    if torch is None:
        raise RuntimeError("torch is required to combine datasets")

    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    test_dataset = test_sets[0] if len(test_sets) == 1 else ConcatDataset(test_sets)
    return train_dataset, val_dataset, test_dataset
