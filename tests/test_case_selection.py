"""Tests for qualitative case selection (no model rerun)."""

from __future__ import annotations

import numpy as np

from inspectai.evaluation.case_selection import (
    flatten_selection,
    select_analysis_indices,
)
from inspectai.visualization.panels import analysis_filename


def test_selects_median_defect_and_score_extremes() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    groups = np.array(
        ["good", "good", "good", "scratch", "scratch", "scratch", "dent", "dent", "dent"]
    )
    scores = np.array([0.9, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 0.1, 2.0])

    selected = select_analysis_indices(labels, groups, scores, n_high_good=2, n_low_defect=2)

    assert selected["per_defect"]["scratch"] == 4
    assert selected["per_defect"]["dent"] == 8
    assert selected["high_score_good"] == [0, 2]
    assert selected["low_score_defect"] == [7, 3]


def test_flatten_and_filename() -> None:
    selected = {
        "per_defect": {"scratch": 1},
        "high_score_good": [0],
        "low_score_defect": [2],
    }
    rows = flatten_selection(selected)
    assert rows[0] == ("per_defect", "scratch", 1)
    name = analysis_filename(
        defect_type="bent_lead",
        label=1,
        score=1.23456,
        image_path="data/raw/transistor/test/bent_lead/003.png",
    )
    assert name == "bent_lead_label1_score1.2346_003.png"
