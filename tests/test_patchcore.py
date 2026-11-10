"""Lightweight PatchCore-style tests. Does not run full MVTec evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from inspectai.models.patchcore import (
    FrozenResNetMultiScaleEncoder,
    PatchCoreConfig,
    PatchCoreScorer,
    align_and_concat,
    deterministic_candidate_pool,
    greedy_farthest_point_indices,
    nearest_neighbor_l2,
)


class TinyFusedEncoder(nn.Module):
    """Two-scale conv tower used as a stand-in for layer2/layer3 fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.high = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        self.low = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = self.high(x)
        low = self.low(high)
        return align_and_concat(high, low)


def test_align_and_concat_spatial_and_channels() -> None:
    high = torch.randn(1, 128, 16, 16)
    low = torch.randn(1, 256, 8, 8)
    fused = align_and_concat(high, low)
    assert fused.shape == (1, 384, 16, 16)
    norms = torch.linalg.vector_norm(fused, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_resnet_layer2_layer3_shapes() -> None:
    pytest.importorskip("torchvision")
    encoder = FrozenResNetMultiScaleEncoder()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        layer2, layer3 = encoder.forward_layers(x)
        fused = encoder(x)
    assert layer2.shape[-2:] == (8, 8)
    assert layer3.shape[-2:] == (4, 4)
    assert layer2.shape[1] == 128
    assert layer3.shape[1] == 256
    assert fused.shape == (1, 384, 8, 8)


def test_candidate_pool_is_deterministic_and_capped() -> None:
    a = deterministic_candidate_pool(80_000, 50_000, seed=42)
    b = deterministic_candidate_pool(80_000, 50_000, seed=42)
    c = deterministic_candidate_pool(80_000, 50_000, seed=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 50_000
    assert len(set(a.tolist())) == 50_000
    small = deterministic_candidate_pool(100, 50_000, seed=42)
    assert len(small) == 100
    assert np.array_equal(small, np.arange(100))


def test_presample_then_fps_is_deterministic_and_capped() -> None:
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((200, 16)).astype(np.float32)
    pool_a = deterministic_candidate_pool(len(raw), 80, seed=42)
    pool_b = deterministic_candidate_pool(len(raw), 80, seed=42)
    assert np.array_equal(pool_a, pool_b)
    fps_a = greedy_farthest_point_indices(raw[pool_a], 25, seed=42, projection_dim=8)
    fps_b = greedy_farthest_point_indices(raw[pool_a], 25, seed=42, projection_dim=8)
    assert np.array_equal(fps_a, fps_b)
    selected = pool_a[fps_a]
    assert len(selected) == 25
    assert len(set(selected.tolist())) == 25


def test_coreset_is_deterministic_and_capped() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((50, 16)).astype(np.float32)
    a = greedy_farthest_point_indices(features, 10, seed=42, projection_dim=8)
    b = greedy_farthest_point_indices(features, 10, seed=42, projection_dim=8)
    c = greedy_farthest_point_indices(features, 10, seed=0, projection_dim=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 10
    assert len(set(a.tolist())) == 10
    all_idx = greedy_farthest_point_indices(features, 200, seed=42, projection_dim=8)
    assert len(all_idx) == 50


def test_nearest_neighbor_and_scorer_cpu() -> None:
    memory = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)
    dist = nearest_neighbor_l2(queries, memory)
    assert dist.shape == (2,)
    assert dist[0] < dist[1] or dist[0] >= 0.0

    cfg = PatchCoreConfig(input_size=16, max_memory_patches=20, projection_dim=8)
    scorer = PatchCoreScorer(TinyFusedEncoder(), config=cfg)
    train = [torch.full((3, 16, 16), 40, dtype=torch.uint8) for _ in range(3)]
    scorer.fit(train, log=None)
    assert scorer.memory is not None
    assert scorer.memory.shape[0] <= 20
    assert scorer.n_pool_patches <= scorer.config.candidate_pool_size
    assert scorer.memory.shape[0] <= scorer.config.max_memory_patches
    score, amap = scorer.score_image(train[0], (16, 16))
    assert isinstance(score, float)
    assert amap.shape == (16, 16)
    meta = scorer.memory_metadata()
    assert meta["n_memory_patches"] <= 20
    assert meta["n_candidate_patches"] >= meta["n_memory_patches"]


def test_rejects_unlocked_k() -> None:
    cfg = PatchCoreConfig(k_neighbors=3)
    with pytest.raises(ValueError, match="k_neighbors=1"):
        PatchCoreScorer(TinyFusedEncoder(), config=cfg)
