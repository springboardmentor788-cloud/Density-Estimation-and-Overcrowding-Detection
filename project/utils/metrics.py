from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def mae(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pred = np.asarray(list(predictions), dtype=np.float64)
    tgt = np.asarray(list(targets), dtype=np.float64)
    if pred.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - tgt)))


def rmse(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pred = np.asarray(list(predictions), dtype=np.float64)
    tgt = np.asarray(list(targets), dtype=np.float64)
    if pred.size == 0:
        return 0.0
    return float(math.sqrt(np.mean((pred - tgt) ** 2)))


def summarize_counts(predictions: list[float], targets: list[float]) -> dict[str, float]:
    return {
        "mae": mae(predictions, targets),
        "rmse": rmse(predictions, targets),
        "pred_mean": float(np.mean(predictions)) if predictions else 0.0,
        "target_mean": float(np.mean(targets)) if targets else 0.0,
    }
