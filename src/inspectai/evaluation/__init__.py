"""Evaluation metrics for anomaly detection."""

from inspectai.evaluation.case_selection import (
    flatten_selection,
    select_analysis_indices,
)
from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group

__all__ = [
    "binary_auroc",
    "flatten_selection",
    "image_auroc_per_group",
    "select_analysis_indices",
]
