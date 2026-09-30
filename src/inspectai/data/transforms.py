"""Paired image/mask transforms.

Spatial transforms must apply the same geometry to ``image`` and ``mask``.
This milestone ships only an identity transform; do not resize or normalize here.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PairedTransform(Protocol):
    """Callable applied after load: ``(image, mask) -> (image, mask)``."""

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class IdentityTransform:
    """Leave image and mask unchanged (native resolution)."""

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return image, mask
