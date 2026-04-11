import cv2
import numpy as np
import scipy.io
from scipy.spatial import KDTree


# ---------------- LOAD IMAGE ---------------- #
def load_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"❌ Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0   # ✅ normalize

    return img


# ---------------- LOAD ANNOTATION ---------------- #
def load_annotation(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]
    except Exception as e:
        raise ValueError(f"❌ Error loading annotation: {mat_path} | {e}")

    return points


# ---------------- GENERATE DENSITY MAP ---------------- #
def generate_density_map(image_shape, points):
    height, width = image_shape[:2]
    density = np.zeros((height, width), dtype=np.float32)

    if len(points) == 0:
        return density

    tree = KDTree(points.copy(), leafsize=2048)
    distances, _ = tree.query(points, k=4)

    for i, point in enumerate(points):

        x = int(min(width - 1, max(0, point[0])))
        y = int(min(height - 1, max(0, point[1])))

        if len(points) > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = 15

        sigma = max(1, sigma)

        size = int(6 * sigma + 1)

        if size % 2 == 0:
            size += 1

        gaussian = cv2.getGaussianKernel(size, sigma)
        gaussian = gaussian @ gaussian.T

        half = size // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(width, x + half + 1)
        y2 = min(height, y + half + 1)

        g_x1 = max(0, half - x)
        g_y1 = max(0, half - y)

        h = y2 - y1
        w = x2 - x1

        g_x2 = g_x1 + w
        g_y2 = g_y1 + h

        gaussian_patch = gaussian[g_y1:g_y2, g_x1:g_x2]

        if gaussian_patch.shape != (h, w):
            gaussian_patch = cv2.resize(gaussian_patch, (w, h))

        density[y1:y2, x1:x2] += gaussian_patch

    return density


# ---------------- GET COUNT ---------------- #
def get_count(density_map):
    return np.sum(density_map)
