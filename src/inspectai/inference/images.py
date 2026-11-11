"""Canonical image conversion for Inspector: uint8 HWC RGB."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage


def to_uint8_rgb(image: np.ndarray | PILImage) -> np.ndarray:
    """Convert a PIL image or numpy array to contiguous uint8 HWC RGB."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
        return np.ascontiguousarray(array, dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC RGB array, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(f"Expected 3 color channels, got shape {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    values = array.astype(np.float64, copy=False)
    if values.max() <= 1.0:
        values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0.0, 255.0).astype(np.uint8))
