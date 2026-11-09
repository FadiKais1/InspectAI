"""Anomaly-map overlays. Image and map must already be aligned spatially."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def anomaly_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend an RGB uint8 image with a normalized heatmap (HW)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    heat = np.asarray(score_map, dtype=np.float64)
    if heat.shape != image.shape[:2]:
        raise ValueError(f"Map {heat.shape} does not match image {image.shape[:2]}")
    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        norm = np.zeros_like(heat)
    else:
        lo, hi = float(finite.min()), float(finite.max())
        norm = (heat - lo) / (hi - lo + 1e-8)
    cmap = np.zeros((*heat.shape, 3), dtype=np.float64)
    cmap[..., 0] = np.clip(norm * 2.0, 0.0, 1.0)
    cmap[..., 1] = np.clip(2.0 - norm * 2.0, 0.0, 1.0)
    rgb = image.astype(np.float64) / 255.0
    blend = (1.0 - alpha) * rgb + alpha * cmap
    return np.clip(blend * 255.0, 0, 255).astype(np.uint8)


def save_rgb(path: str | Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path)
