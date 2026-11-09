"""Unit tests for patch-kNN using a tiny encoder (no torchvision, no MVTec)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from inspectai.evaluation.metrics import binary_auroc
from inspectai.models.patch_knn import PatchKNNConfig, PatchKNNScorer
from inspectai.visualization.overlays import anomaly_overlay


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(x), p=2, dim=1)


def _solid(value: int, size: int = 16) -> torch.Tensor:
    return torch.full((3, size, size), value, dtype=torch.uint8)


def test_memory_fits_only_on_provided_train_images() -> None:
    cfg = PatchKNNConfig(input_size=16, max_memory_patches=10_000)
    scorer = PatchKNNScorer(TinyEncoder(), config=cfg)
    train = [_solid(10), _solid(12)]
    scorer.fit(train)
    assert scorer.memory is not None
    assert scorer.memory.ndim == 2
    assert scorer.memory.shape[0] > 0


def test_identical_image_scores_lower_than_different_image() -> None:
    cfg = PatchKNNConfig(input_size=16, max_memory_patches=10_000)
    scorer = PatchKNNScorer(TinyEncoder(), config=cfg)
    train = [_solid(40), _solid(42)]
    scorer.fit(train)
    same, _ = scorer.score_image(_solid(40), (16, 16))
    different, score_map = scorer.score_image(_solid(200), (16, 16))
    assert score_map.shape == (16, 16)
    assert different >= same


def test_overlay_shape_matches_image() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    heat = np.linspace(0, 1, 64, dtype=np.float64).reshape(8, 8)
    out = anomaly_overlay(image, heat)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_auroc_on_synthetic_scores() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([1.0, 1.1, 3.0, 2.5])
    assert binary_auroc(y, s) == 1.0


def test_rejects_k_not_one() -> None:
    cfg = PatchKNNConfig(k_neighbors=3, input_size=16)
    with pytest.raises(ValueError, match="k_neighbors=1"):
        PatchKNNScorer(TinyEncoder(), config=cfg)
