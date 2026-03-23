import os
import cv2
import numpy as np
from config import DATASET_PATH, RESIZE_WIDTH, RESIZE_HEIGHT, NORMALIZE


def load_dataset(folder):
    images = []

    files = os.listdir(folder)
    print("Total images found:", len(files))

    for i, file in enumerate(files):
        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        # 🔹 Show original size
        original_shape = img.shape

        # Resize image
        img_resized = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
        resized_shape = img_resized.shape

        # Normalize if required
        if NORMALIZE:
            img_resized = img_resized / 255.0

        images.append(img_resized)

        # Print info only for first 5 images (to avoid huge output)
        if i < 5:
            print(f"\nImage: {file}")
            print("Original size:", original_shape)
            print("Resized size:", resized_shape)

    return np.array(images)


# Run preprocessing
data = load_dataset(DATASET_PATH)

print("\nFinal dataset shape:", data.shape)