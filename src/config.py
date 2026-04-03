from pathlib import Path


# Base project directory (assumes this file lives in src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw dataset root (you already have the dataset; just point here)
DATA_ROOT = PROJECT_ROOT / "data" / "ShanghaiTech"

# Preprocessed dataset root
PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "ShanghaiTech_preprocessed"


class PreprocessConfig:
    """
    Centralized configuration for preprocessing.
    """

    # (width, height) for resizing images
    TARGET_SIZE = (512, 512)

    # File extensions considered as input images
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "PREPROCESSED_ROOT",
    "PreprocessConfig",
]

