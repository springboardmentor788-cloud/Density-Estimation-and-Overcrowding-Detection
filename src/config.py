from pathlib import Path

BASE_DIR = Path(r"C:\Users\deepa\OneDrive\Documents\OneDrive\Desktop\virtual internship")


DATA_ROOT         = BASE_DIR / "dataset" / "archive"

PART_A_TRAIN      = DATA_ROOT / "part_A_final" / "train_data" / "images"
PART_A_TEST       = DATA_ROOT / "part_A_final" / "test_data"  / "images"
PART_B_TRAIN      = DATA_ROOT / "part_B_final" / "train_data" / "images"
PART_B_TEST       = DATA_ROOT / "part_B_final" / "test_data"  / "images"

PREPROCESSED_ROOT = BASE_DIR / "data" / "preprocessed"
OUTPUT_DIR        = BASE_DIR / "outputs"
MODEL_DIR         = BASE_DIR / "models"

IMAGE_SIZE   = (224, 224)
CHANNELS     = 3
PARTS        = ["A", "B"]
SPLITS       = ["train", "test"]
BATCH_SIZE   = 8
NUM_WORKERS  = 0
PIN_MEMORY   = False
SEED         = 42
VERBOSE      = True
BATCH_SIZE   = 16    # change 8 to any number you want