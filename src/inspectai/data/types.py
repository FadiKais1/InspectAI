"""Public data-pipeline types and errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MVTecError(Exception):
    """Base error for MVTec layout, pairing, or loading failures."""


class MVTecLayoutError(MVTecError):
    """The on-disk category layout is missing or invalid for the requested split."""


class MVTecPairingError(MVTecError):
    """Image/mask pairing or spatial alignment is invalid."""


@dataclass
class MVTecSample:
    """One native-resolution MVTec sample (NumPy, not tensors)."""

    image: np.ndarray
    mask: np.ndarray
    label: int
    defect_type: str
    image_path: Path
