from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy import io
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def parse_gt_points(mat_path: Path) -> np.ndarray:
    """Load ShanghaiTech point annotations from .mat and return Nx2 (x, y)."""
    mat = io.loadmat(str(mat_path))
    points = mat["image_info"][0, 0][0, 0][0]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return points


def adaptive_gaussian_density(
    image_shape: Tuple[int, int],
    points_xy: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Build adaptive-kernel density map. Sum is approximately number of points."""
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points_xy) == 0:
        return density

    points = points_xy.copy()
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)

    if len(points) > 1:
        tree = KDTree(points, leafsize=2048)
        distances, _ = tree.query(points, k=min(k, len(points)))

    for i, (x, y) in enumerate(points):
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        impulse = np.zeros((h, w), dtype=np.float32)
        impulse[y_i, x_i] = 1.0

        if len(points) > 1:
            neigh = distances[i][1:] if distances.ndim > 1 else np.array([distances[i]], dtype=np.float32)
            sigma = float(np.mean(neigh) * 0.3)
            sigma = max(1.0, sigma)
        else:
            sigma = float((h + w) / 4.0)

        blurred = gaussian_filter(impulse, sigma=sigma, mode="constant")
        mass = float(blurred.sum())
        if mass > 0:
            density += blurred / mass

    return density


def downsample_density_map(density: np.ndarray, output_stride: int = 8) -> np.ndarray:
    """Downsample map while preserving integral count."""
    original_count = float(density.sum())
    h, w = density.shape
    new_h = max(1, h // output_stride)
    new_w = max(1, w // output_stride)
    down = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_AREA)
    down = down.astype(np.float32) * float(output_stride * output_stride)
    new_count = float(down.sum())
    if new_count > 0:
        down *= original_count / new_count
    return down


def load_or_create_density_map(
    image_path: Path,
    mat_path: Path,
    image_shape: Tuple[int, int],
    cache_root: Path,
    output_stride: int = 8,
    force_rebuild: bool = False,
    point_scale: Tuple[float, float] = (1.0, 1.0),
    cache_version: str = "v2",
) -> np.ndarray:
    """Read density from cache or generate and cache it as .npy."""
    cache_root.mkdir(parents=True, exist_ok=True)
    h, w = image_shape
    cache_file = cache_root / f"{image_path.stem}_{h}x{w}_os{output_stride}_{cache_version}.npy"

    if cache_file.exists() and not force_rebuild:
        return np.load(cache_file).astype(np.float32)

    points = parse_gt_points(mat_path)
    sx, sy = point_scale
    if len(points) > 0 and (sx != 1.0 or sy != 1.0):
        points = points.copy()
        points[:, 0] *= float(sx)
        points[:, 1] *= float(sy)
    density_full = adaptive_gaussian_density(image_shape=image_shape, points_xy=points)
    density = downsample_density_map(density_full, output_stride=output_stride)

    np.save(cache_file, density.astype(np.float32))
    return density.astype(np.float32)
