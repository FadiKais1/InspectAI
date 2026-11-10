"""Select qualitative analysis cases from already-computed test scores.

Uses labels only for visualization grouping, not for fitting or metric design.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _median_index(indices: np.ndarray, scores: np.ndarray) -> int:
    order = indices[np.argsort(scores[indices], kind="mergesort")]
    return int(order[len(order) // 2])


def select_analysis_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    *,
    n_high_good: int = 3,
    n_low_defect: int = 3,
    normal_group: str = "good",
) -> dict[str, list[int]]:
    """Return index lists for per-defect, high-score normals, and low-score defects."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    scores = np.asarray(scores, dtype=np.float64)
    if not (labels.shape == groups.shape == scores.shape):
        raise ValueError("labels, groups, and scores must have the same length")

    per_defect: dict[str, int] = {}
    defect_names = sorted({str(g) for g in groups if str(g) != normal_group})
    for name in defect_names:
        idxs = np.flatnonzero((groups == name) & (labels == 1))
        if idxs.size == 0:
            continue
        per_defect[name] = _median_index(idxs, scores)

    good_idxs = np.flatnonzero(labels == 0)
    high_good = [
        int(i)
        for i in good_idxs[np.argsort(-scores[good_idxs], kind="mergesort")][:n_high_good]
    ]

    defect_idxs = np.flatnonzero(labels == 1)
    low_defect = [
        int(i)
        for i in defect_idxs[np.argsort(scores[defect_idxs], kind="mergesort")][:n_low_defect]
    ]

    return {
        "per_defect": per_defect,
        "high_score_good": high_good,
        "low_score_defect": low_defect,
    }


def flatten_selection(selection: Mapping[str, object]) -> list[tuple[str, str, int]]:
    """Flatten selection into (bucket, tag, index) rows for saving figures."""
    rows: list[tuple[str, str, int]] = []
    per_defect = selection["per_defect"]
    assert isinstance(per_defect, dict)
    for name in sorted(per_defect):
        rows.append(("per_defect", str(name), int(per_defect[name])))
    for i, idx in enumerate(selection["high_score_good"]):  # type: ignore[arg-type]
        rows.append(("high_score_good", f"rank{i + 1}", int(idx)))
    for i, idx in enumerate(selection["low_score_defect"]):  # type: ignore[arg-type]
        rows.append(("low_score_defect", f"rank{i + 1}", int(idx)))
    return rows
