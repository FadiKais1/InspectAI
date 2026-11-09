"""Unit tests for AUROC helpers."""

from __future__ import annotations

import numpy as np

from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group


def test_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(y, s) == 1.0


def test_inverted_scores() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert binary_auroc(y, s) == 0.0


def test_missing_class_is_nan() -> None:
    y = np.zeros(4)
    s = np.arange(4.0)
    assert np.isnan(binary_auroc(y, s))


def test_per_defect_uses_normals_plus_one_group() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.3])
    groups = np.array(["good", "good", "scratch", "dent"])
    out = image_auroc_per_group(labels, scores, groups)
    assert set(out) == {"scratch", "dent"}
    assert out["scratch"] == 1.0
