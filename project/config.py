from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class CrowdConfig:
    project_root: Path = PROJECT_ROOT
    data_roots: tuple[Path, ...] = (PROJECT_ROOT / "part_A_final", PROJECT_ROOT / "part_B_final")
    cache_dir: Path = PROJECT_ROOT / "cache" / "density_maps"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"
    logs_dir: Path = PROJECT_ROOT / "logs"

    train_split: float = 0.8
    val_split: float = 0.2
    random_seed: int = 42

    image_size: tuple[int, int] = (384, 384)
    crop_size: tuple[int, int] | None = (256, 256)
    batch_size: int = 8
    num_workers: int = 2

    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip_norm: float = 5.0
    use_amp: bool = False
    count_loss_weight: float = 1e-4
    count_loss_weight_max: float = 5e-4

    density_sigma_scale: float = 0.3
    density_min_sigma: float = 4.0
    density_knn: int = 4

    inference_resize_width: int = 640
    video_sample_fps: float = 2.0
    alert_threshold: float = 120.0
    alert_cooldown_frames: int = 15

    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def dataset_root(self, name: str) -> Path:
        mapping = {
            "a": self.project_root / "part_A_final",
            "b": self.project_root / "part_B_final",
            "part_a": self.project_root / "part_A_final",
            "part_b": self.project_root / "part_B_final",
        }
        key = name.lower()
        return mapping.get(key, self.project_root / name)

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.checkpoint_dir, self.output_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


CONFIG = CrowdConfig()
