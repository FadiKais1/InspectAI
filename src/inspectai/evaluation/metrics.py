"""Binary AUROC helpers. Do not invent scores; call these on real predictions."""

from __future__ import annotations

import numpy as np


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC. Returns NaN if a class is missing."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch: labels {y.shape} vs scores {s.shape}")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    # Average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and sorted_s[j] == sorted_s[i]:
            j += 1
        if j > i + 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[y].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def image_auroc_per_group(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    normal_group: str = "good",
) -> dict[str, float]:
    """Image AUROC for each anomalous group against all normal samples."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    out: dict[str, float] = {}
    normal = groups == normal_group
    defect_names = sorted({str(g) for g in groups if str(g) != normal_group})
    for name in defect_names:
        mask = normal | (groups == name)
        out[name] = binary_auroc(labels[mask], scores[mask])
    return out
