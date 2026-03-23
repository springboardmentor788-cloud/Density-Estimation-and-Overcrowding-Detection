import os
import cv2
import matplotlib.pyplot as plt

folder = "dataset/part_A/test_data/images"

images = os.listdir(folder)

print("Total images found:", len(images))

# Show first 5 images
for i in range(5):
    path = os.path.join(folder, images[i])
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(images[i])
    plt.axis("off")

plt.show()