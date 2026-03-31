from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetConfig:
    root: Path
    part: str = "A"
    split: str = "train"
    max_dim: int = 1536
    crop_size: int = 512
    output_stride: int = 8
    cache_dir: Path | None = None


@dataclass
class TrainConfig:
    batch_size: int = 4
    workers: int = 8
    epochs: int = 300
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    seed: int = 42


@dataclass
class InferenceConfig:
    max_dim: int = 1280
    use_fp16: bool = True
    overcrowd_count_threshold: int = 120
    region_threshold: float = 0.0025
    alert_cooldown_sec: float = 2.0
