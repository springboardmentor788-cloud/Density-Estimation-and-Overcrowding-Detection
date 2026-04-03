"""
Utilities for generating ground-truth density maps from point annotations.
Used for ShanghaiTech and similar crowd counting datasets.
"""

import numpy as np


def gaussian_kernel_2d(size: int = 15, sigma: float = 2.0) -> np.ndarray:
    """Create a 2D Gaussian kernel (sums to 1)."""
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def points_to_density_map(
    points: np.ndarray,
    out_h: int,
    out_w: int,
    kernel_size: int = 15,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Generate a density map from head annotations.

    Args:
        points: (N, 2) array, each row (x, y) in pixel coords (in density map space).
        out_h, out_w: Height and width of the density map.
        kernel_size: Size of Gaussian kernel.
        sigma: Sigma of Gaussian.

    Returns:
        (out_h, out_w) float array; sum equals number of points (each head contributes 1).
    """
    density = np.zeros((out_h, out_w), dtype=np.float32)
    if points.size == 0:
        return density

    # Handle (N,) or (N,1) or (2,N) etc.
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[1] != 2:
        pts = pts.T
    if pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)

    h, w = out_h, out_w
    half = kernel_size // 2
    kernel = gaussian_kernel_2d(kernel_size, sigma)

    for i in range(pts.shape[0]):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)

        y1, y2 = max(0, y - half), min(h, y + half + 1)
        x1, x2 = max(0, x - half), min(w, x + half + 1)

        ky1, ky2 = half - (y - y1), half + (y2 - y)
        kx1, kx2 = half - (x - x1), half + (x2 - x)

        k = kernel[half - (y - y1) : half + (y2 - y), half - (x - x1) : half + (x2 - x)]
        density[y1:y2, x1:x2] += k

    return density


def crop_kernel(kernel: np.ndarray, center: int, length: int) -> tuple[np.ndarray, int, int]:
    """Return a cropped kernel and the start index for placing it (for edge handling)."""
    half = kernel.shape[0] // 2
    start = center - half
    end = start + kernel.shape[0]
    if start < 0:
        kernel = kernel[-start:, :]
        start = 0
    if end > length:
        kernel = kernel[: length - end, :]
    return kernel, start, min(end, length)


def points_to_density_map_simple(
    points: np.ndarray,
    out_h: int,
    out_w: int,
    kernel_size: int = 15,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Simpler implementation: place a Gaussian of sum=1 at each point.
    Clips coordinates to valid range.
    """
    density = np.zeros((out_h, out_w), dtype=np.float32)
    if points.size == 0:
        return density

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[1] != 2:
        pts = pts.T
    if pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)

    half = kernel_size // 2
    kernel = gaussian_kernel_2d(kernel_size, sigma)

    for i in range(pts.shape[0]):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        y_lo = max(0, y - half)
        y_hi = min(out_h, y + half + 1)
        x_lo = max(0, x - half)
        x_hi = min(out_w, x + half + 1)

        k_y_lo = y_lo - (y - half)
        k_y_hi = k_y_lo + (y_hi - y_lo)
        k_x_lo = x_lo - (x - half)
        k_x_hi = k_x_lo + (x_hi - x_lo)

        patch = kernel[k_y_lo:k_y_hi, k_x_lo:k_x_hi]
        density[y_lo:y_hi, x_lo:x_hi] += patch

    return density


__all__ = ["gaussian_kernel_2d", "points_to_density_map", "points_to_density_map_simple"]
