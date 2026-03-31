from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from project.config import IMAGENET_MEAN, IMAGENET_STD
from project.models.csrnet import CSRNet
from project.models.mcnn import MCNN


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.times = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now)
        if len(self.times) < 2:
            return 0.0
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / max(elapsed, 1e-8)


def build_model(name: str, use_batch_norm: bool = False) -> torch.nn.Module:
    if name.lower() == "csrnet":
        return CSRNet(load_pretrained_frontend=False, use_batch_norm=use_batch_norm)
    if name.lower() == "mcnn":
        return MCNN()
    raise ValueError(f"Unsupported model type: {name}")


def load_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    use_batch_norm: bool = False,
) -> torch.nn.Module:
    model = build_model(model_name, use_batch_norm=use_batch_norm)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def resize_frame_max_dim(frame_bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale == 1.0:
        return frame_bgr
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def preprocess_frame(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = frame_rgb.astype(np.float32) / 255.0
    image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(image).float().to(device)
    return tensor
