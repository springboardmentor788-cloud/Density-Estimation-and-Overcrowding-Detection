import os
import random
import shutil

dataset_path = "dataset/ShanghaiTech/part_A"
train_path = "dataset_split/train"
val_path = "dataset_split/val"
test_path = "dataset_split/test"

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

images = os.listdir(dataset_path)
random.shuffle(images)

train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

train_imgs = images[:train_split]
val_imgs = images[train_split:val_split]
test_imgs = images[val_split:]

for img in train_imgs:
    shutil.copy(os.path.join(dataset_path,img), os.path.join( train_path,img))

for img in val_imgs:
    shutil.copy(os.path.join(dataset_path,img), val_path)

for img in test_imgs:
    shutil.copy(os.path.join(dataset_path,img), test_path)

print("Dataset Split Complete")