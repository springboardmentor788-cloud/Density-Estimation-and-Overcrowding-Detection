from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _normalize_map(density: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=np.float32)
    max_value = float(density.max()) if density.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(density, dtype=np.uint8)
    return np.clip(density / max_value * 255.0, 0, 255).astype(np.uint8)


def density_to_heatmap(density: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for heatmap visualization")
    normalized = _normalize_map(density)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def overlay_heatmap(image_bgr: np.ndarray, density: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for overlay visualization")
    heatmap = density_to_heatmap(density)
    if heatmap.shape[:2] != image_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def _panel(title: str, image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for visualization panels")
    panel = image_bgr.copy()
    if panel.shape[:2] != size:
        panel = cv2.resize(panel, (size[1], size[0]))
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (255, 255, 255), 1)
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def save_comparison_figure(
    image_bgr: np.ndarray,
    gt_density: np.ndarray,
    pred_density: np.ndarray,
    *,
    gt_count: float,
    pred_count: float,
    output_path: str | Path,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for visualization exports")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_panel = _panel(f"Input | GT {gt_count:.1f} | Pred {pred_count:.1f}", image_bgr, (360, 480))
    gt_panel = _panel("Ground Truth Density", density_to_heatmap(gt_density), (360, 480))
    pred_panel = _panel("Predicted Density", density_to_heatmap(pred_density), (360, 480))
    blank = np.zeros_like(pred_panel)

    top = np.concatenate([image_panel, gt_panel], axis=1)
    bottom = np.concatenate([pred_panel, blank], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    cv2.imwrite(str(output_path), canvas)
