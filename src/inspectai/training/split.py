"""Deterministic train/validation index splits (normal images only)."""

from __future__ import annotations

import numpy as np


def train_val_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute ``0..n-1`` with ``seed`` and cut at ``floor(n * train_fraction)``."""
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to split train/validation")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * train_fraction)
    if n_train < 1 or n_train >= n_samples:
        raise ValueError("Split would leave an empty train or validation set")
    return perm[:n_train].astype(np.int64), perm[n_train:].astype(np.int64)
