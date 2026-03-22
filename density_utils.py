import numpy as np
from scipy.ndimage import gaussian_filter


def generate_density_map(img_shape, points):

    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for p in points:
        x = min(w - 1, int(p[0]))
        y = min(h - 1, int(p[1]))
        density[y, x] = 1

    # ✅ Adaptive sigma (FIXED INDENTATION)
    if len(points) < 50:
        sigma = 2
    elif len(points) < 200:
        sigma = 3
    else:
        sigma = 4

    density = gaussian_filter(density, sigma=sigma)

    return density