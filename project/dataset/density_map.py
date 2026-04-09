from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.nn import functional as F
except Exception:  # pragma: no cover - import guard for environments without torch
    torch = None
    F = None

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover - scipy is a runtime dependency
    loadmat = None


def _candidate_point_arrays(value: object) -> Iterable[np.ndarray]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _candidate_point_arrays(item)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                yield from _candidate_point_arrays(value[field])
            return

        if value.dtype == object:
            for item in value.flat:
                yield from _candidate_point_arrays(item)
            return

        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[-1] == 2 and array.size >= 2:
            reshaped = array.reshape(-1, 2)
            yield reshaped.astype(np.float32)
        return

    if hasattr(value, "__dict__"):
        for item in vars(value).values():
            yield from _candidate_point_arrays(item)


def load_points_from_mat(mat_path: str | Path) -> np.ndarray:
    if loadmat is None:
        raise RuntimeError("scipy is required to read ShanghaiTech .mat annotations")

    data = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    candidates: list[np.ndarray] = []

    if "image_info" in data:
        candidates.extend(_candidate_point_arrays(data["image_info"]))
    candidates.extend(_candidate_point_arrays(data))

    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.ndim == 2 and candidate.shape[1] == 2 and candidate.shape[0] >= 0:
            return candidate

    raise ValueError(f"No point annotations found in {mat_path}")


@lru_cache(maxsize=4096)
def load_points_from_mat_cached(mat_path: str) -> np.ndarray:
    return load_points_from_mat(mat_path)


def points_to_count(points: np.ndarray) -> float:
    return float(points.shape[0])


def compute_adaptive_sigmas(
    points: np.ndarray,
    *,
    knn: int = 4,
    min_sigma: float = 4.0,
    sigma_scale: float = 0.3,
    max_sigma: float | None = None,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if points.shape[0] == 1:
        sigma = np.array([max(min_sigma, 1.0)], dtype=np.float32)
        if max_sigma is not None:
            sigma = np.clip(sigma, min_sigma, max_sigma)
        return sigma

    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    k = max(1, min(knn, points.shape[0] - 1))
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    sigma = np.mean(nearest, axis=1) * sigma_scale
    sigma = np.maximum(sigma, min_sigma)
    if max_sigma is not None:
        sigma = np.clip(sigma, min_sigma, max_sigma)
    return sigma.astype(np.float32)


@lru_cache(maxsize=2048)
def _gaussian_kernel(radius: int, sigma_key: int) -> np.ndarray:
    sigma = max(sigma_key / 100.0, 0.5)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return np.zeros_like(kernel, dtype=np.float32)
    return (kernel / kernel_sum).astype(np.float32)


def generate_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    *,
    knn: int = 4,
    sigma_scale: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float | None = None,
) -> np.ndarray:
    height, width = image_shape
    density = np.zeros((height, width), dtype=np.float32)
    if points.size == 0:
        return density

    sigmas = compute_adaptive_sigmas(
        points,
        knn=knn,
        sigma_scale=sigma_scale,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )

    for (x_coord, y_coord), sigma in zip(points, sigmas, strict=False):
        x = int(np.clip(round(float(x_coord)), 0, width - 1))
        y = int(np.clip(round(float(y_coord)), 0, height - 1))

        radius = max(1, int(np.ceil(3.0 * float(sigma))))
        kernel = _gaussian_kernel(radius, int(round(float(sigma) * 100.0)))

        x0 = max(0, x - radius)
        x1 = min(width - 1, x + radius)
        y0 = max(0, y - radius)
        y1 = min(height - 1, y + radius)

        kernel_x0 = radius - (x - x0)
        kernel_x1 = radius + (x1 - x)
        kernel_y0 = radius - (y - y0)
        kernel_y1 = radius + (y1 - y)

        clipped_kernel = kernel[kernel_y0 : kernel_y1 + 1, kernel_x0 : kernel_x1 + 1]
        clipped_sum = float(clipped_kernel.sum())
        if clipped_sum > 0.0:
            clipped_kernel = clipped_kernel / clipped_sum
        density[y0 : y1 + 1, x0 : x1 + 1] += clipped_kernel

    return density


def density_cache_path(cache_root: str | Path, sample_id: str, shape: tuple[int, int]) -> Path:
    height, width = shape
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{sample_id}_{height}x{width}.npy"


def load_density_cache(cache_path: str | Path) -> np.ndarray | None:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    return np.load(cache_path)


def save_density_cache(cache_path: str | Path, density: np.ndarray) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, density.astype(np.float32))


def load_or_create_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    cache_path: str | Path | None = None,
    *,
    knn: int = 4,
    sigma_scale: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float | None = None,
    force_rebuild: bool = False,
) -> np.ndarray:
    if cache_path is not None and not force_rebuild:
        cached = load_density_cache(cache_path)
        if cached is not None:
            return cached.astype(np.float32)

    density = generate_density_map(
        image_shape,
        points,
        knn=knn,
        sigma_scale=sigma_scale,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )
    if cache_path is not None:
        save_density_cache(cache_path, density)
    return density


def resize_density_tensor(density: "torch.Tensor", target_hw: tuple[int, int]) -> "torch.Tensor":
    if torch is None or F is None:
        raise RuntimeError("torch is required for density resizing")

    if density.ndim == 2:
        density = density.unsqueeze(0).unsqueeze(0)
    elif density.ndim == 3:
        density = density.unsqueeze(0)

    in_h, in_w = density.shape[-2:]
    if (in_h, in_w) == target_hw:
        return density

    resized = F.interpolate(density.float(), size=target_hw, mode="area")
    scale = (in_h * in_w) / float(target_hw[0] * target_hw[1])
    return resized * scale


def density_to_count(density: np.ndarray) -> float:
    return float(np.asarray(density, dtype=np.float64).sum())
