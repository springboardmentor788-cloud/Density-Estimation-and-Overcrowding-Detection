from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _extract_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        candidate_keys = (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        )
        for key in candidate_keys:
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        if raw and all(isinstance(k, str) for k in raw.keys()):
            return raw  # already a plain state dict
    raise ValueError("Unsupported checkpoint format for external weight loading")


def _normalize_key(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "net.",
    )
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _map_legacy_csrnet_key(key: str) -> str:
    frontend_map = {
        "frontend.0.": "frontend.0.0.",
        "frontend.2.": "frontend.1.0.",
        "frontend.5.": "frontend.3.0.",
        "frontend.7.": "frontend.4.0.",
        "frontend.10.": "frontend.6.0.",
        "frontend.12.": "frontend.7.0.",
        "frontend.14.": "frontend.8.0.",
        "frontend.17.": "frontend.10.0.",
        "frontend.19.": "frontend.11.0.",
        "frontend.21.": "frontend.12.0.",
    }
    backend_map = {
        "backend.0.": "backend.0.",
        "backend.2.": "backend.2.",
        "backend.4.": "backend.4.",
        "backend.6.": "backend.6.",
        "backend.8.": "backend.8.",
        "backend.10.": "backend.10.",
        "backend.12.": "backend.12.",
        "output_layer.": "backend.12.",
    }

    for legacy_prefix, target_prefix in frontend_map.items():
        if key.startswith(legacy_prefix):
            return key.replace(legacy_prefix, target_prefix, 1)

    for legacy_prefix, target_prefix in backend_map.items():
        if key.startswith(legacy_prefix):
            return key.replace(legacy_prefix, target_prefix, 1)

    return key


def load_external_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    raw = torch.load(str(checkpoint_path), map_location=map_location)
    state_dict = _extract_state_dict(raw)
    target_keys = set(model.state_dict().keys())

    normalized_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = _normalize_key(key)
        mapped_key = _map_legacy_csrnet_key(normalized_key)
        if mapped_key in target_keys:
            normalized_state[mapped_key] = value
        elif normalized_key in target_keys:
            normalized_state[normalized_key] = value

    load_info = model.load_state_dict(normalized_state, strict=False)
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    return missing, unexpected
