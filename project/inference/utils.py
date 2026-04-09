from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from config import CONFIG
from models.csrnet import CSRNet


@dataclass(slots=True)
class CrowdAlertState:
    threshold: float
    cooldown_frames: int = 15
    _last_alert_frame: int = -10_000
    _last_status: str = "SAFE"

    def update(self, count: float, frame_index: int) -> tuple[str, bool]:
        status = "OVERCROWDED" if count >= self.threshold else "SAFE"
        should_alert = False
        if status == "OVERCROWDED":
            if self._last_status != status or (frame_index - self._last_alert_frame) >= self.cooldown_frames:
                should_alert = True
                self._last_alert_frame = frame_index
        self._last_status = status
        return status, should_alert


@dataclass(slots=True)
class CountCalibration:
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, count: float) -> float:
        return max(0.0, float(self.slope * count + self.intercept))


def load_count_calibration(path: str | Path | None) -> CountCalibration | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return CountCalibration(
        slope=float(payload.get("slope", 1.0)),
        intercept=float(payload.get("intercept", 0.0)),
    )


def get_device(prefer_cuda: bool = True) -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is required for inference")
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str | Path | None, device: "torch.device", *, pretrained_frontend: bool = False) -> CSRNet:
    model = CSRNet(pretrained=pretrained_frontend)
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return model


def resize_frame_keep_aspect(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for inference resizing")
    height, width = frame_bgr.shape[:2]
    if width == target_width:
        return frame_bgr
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def frame_to_tensor(frame_bgr: np.ndarray, device: "torch.device") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is required for inference")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else frame_bgr[..., ::-1]
    image = frame_rgb.astype(np.float32) / 255.0
    image = (image - np.asarray(CONFIG.image_mean, dtype=np.float32)) / np.asarray(CONFIG.image_std, dtype=np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def density_to_count(density_map: np.ndarray) -> float:
    return float(np.asarray(density_map, dtype=np.float32).sum())


def density_to_heatmap(density_map: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for heatmap generation")
    density = np.asarray(density_map, dtype=np.float32)
    max_value = float(density.max()) if density.size else 0.0
    if max_value <= 0.0:
        normalized = np.zeros_like(density, dtype=np.uint8)
    else:
        normalized = np.clip(density / max_value * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def overlay_density(frame_bgr: np.ndarray, density_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for overlay generation")
    heatmap = density_to_heatmap(density_map)
    if heatmap.shape[:2] != frame_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def annotate_frame(frame_bgr: np.ndarray, count: float, status: str, *, alert: bool = False) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for overlay annotation")
    output = frame_bgr.copy()
    accent = (0, 200, 0) if status == "SAFE" else (0, 165, 255)
    if alert:
        accent = (0, 0, 255)

    height, width = output.shape[:2]
    panel_w = min(width - 32, 560)
    panel_h = 92
    x0 = 16
    y0 = max(16, height - panel_h - 16)
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    # Semi-transparent glass panel.
    overlay = output.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 18, 32), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), accent, 2)
    cv2.rectangle(overlay, (x0, y0), (x0 + 10, y1), accent, -1)
    cv2.addWeighted(overlay, 0.58, output, 0.42, 0.0, output)

    # Text hierarchy.
    cv2.putText(output, "CROWD COUNT", (x0 + 22, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 235, 245), 1, cv2.LINE_AA)
    cv2.putText(output, f"{count:.1f}", (x0 + 20, y0 + 68), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

    # Status chip on the right.
    pill_w = 210
    pill_h = 30
    pill_x1 = x1 - 18
    pill_x0 = pill_x1 - pill_w
    pill_y0 = y0 + 20
    pill_y1 = pill_y0 + pill_h
    chip_fill = (24, 32, 48)
    cv2.rectangle(output, (pill_x0, pill_y0), (pill_x1, pill_y1), chip_fill, -1)
    cv2.rectangle(output, (pill_x0, pill_y0), (pill_x1, pill_y1), accent, 1)
    cv2.circle(output, (pill_x0 + 16, pill_y0 + pill_h // 2), 6, accent, -1)
    cv2.putText(output, status, (pill_x0 + 32, pill_y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 250, 255), 1, cv2.LINE_AA)

    if alert:
        alert_text = "OVERCROWDING ALERT"
        cv2.putText(output, alert_text, (x0 + 22, y0 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, accent, 1, cv2.LINE_AA)
    return output


def predict_frame(model: CSRNet, frame_bgr: np.ndarray, device: "torch.device") -> tuple[np.ndarray, float]:
    if torch is None:
        raise RuntimeError("torch is required for inference")
    with torch.inference_mode():
        tensor = frame_to_tensor(frame_bgr, device)
        density = model(tensor).detach().float().cpu().squeeze(0).squeeze(0).numpy()
    return density, density_to_count(density)
