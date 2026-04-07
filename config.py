import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# Dataset Paths
# ----------------------------
DATASET_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "ShanghaiTech",
    "part_A",
    "train_data"
)

IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
GT_DIR = os.path.join(DATASET_ROOT, "ground-truth")

# ----------------------------
# Video Inference
# ----------------------------
VIDEO_PATH = os.path.join(
    BASE_DIR,
    "data",
    "archive",
    "pexels_videos_2740 (1080p).mp4"
)

OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output_density_video.mp4")

# ----------------------------
# Image & Model Settings
# ----------------------------
IMAGE_SIZE = 256          # training size
INFERENCE_SIZE = 640      # video processing size
DOWNSAMPLE = 8
DENSITY_SIZE = IMAGE_SIZE // DOWNSAMPLE

# ----------------------------
# Crowd Monitoring
# ----------------------------
CROWD_THRESHOLD = 100     # 🔥 adjust per video

# ----------------------------
# Training Settings
# ----------------------------
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-5

# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda"