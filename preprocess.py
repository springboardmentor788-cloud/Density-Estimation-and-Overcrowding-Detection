import cv2
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_annotation(mat_path):
    mat = scipy.io.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]
    return points

def generate_density_map(image_shape, points, sigma=4):
    height, width = image_shape[0], image_shape[1]

    density = np.zeros((height, width), dtype=np.float32)

    for point in points:
        x = min(width - 1, int(point[0]))
        y = min(height - 1, int(point[1]))
        density[y, x] = 1

    density = gaussian_filter(density, sigma=sigma)
    return density