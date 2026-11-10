"""Lightweight DINOv3 patch-AD tests. Does not run full MVTec evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from inspectai.models.dinov3_anomaly import (
    OFFICIAL_DINOV3_VITS16,
    DINOv3AnomalyConfig,
    DINOv3PatchScorer,
    DINOWeightsError,
    FrozenDINOv3PatchEncoder,
    dinov3_access_instructions,
    l2_normalize_embeddings,
    load_official_dinov3,
    patch_tokens_to_grid,
    upsample_anomaly_map,
)
from inspectai.models.patchcore import deterministic_candidate_pool, nearest_neighbor_l2


class MockProcessor:
    def __call__(self, images, return_tensors="pt", do_resize=False):
        arr = np.asarray(images, dtype=np.float32) / 255.0
        pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": pixel}


class TinyPatchViT(nn.Module):
    """CPU stand-in: CLS + 4 registers + spatial patches from avg-pooled pixels."""

    def __init__(self, patch_size: int = 16, dim: int = 8, num_register_tokens: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.config = SimpleNamespace(num_register_tokens=num_register_tokens)
        self.proj = nn.Conv2d(3, dim, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor | None = None, **_kwargs):
        assert pixel_values is not None
        pooled = torch.nn.functional.avg_pool2d(pixel_values, kernel_size=self.patch_size)
        patches = self.proj(pooled)
        batch, dim, grid_h, grid_w = patches.shape
        flat = patches.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, dim)
        cls = torch.zeros(batch, 1, dim, device=pixel_values.device, dtype=pixel_values.dtype)
        registers = torch.zeros(
            batch,
            self.config.num_register_tokens,
            dim,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        hidden = torch.cat([cls, registers, flat], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_patch_tokens_to_spatial_grid_drops_cls_and_registers() -> None:
    batch, dim, grid_h, grid_w, registers = 2, 6, 4, 4, 4
    n_patches = grid_h * grid_w
    cls = torch.ones(batch, 1, dim)
    reg = torch.full((batch, registers, dim), 2.0)
    patches = torch.arange(batch * n_patches * dim, dtype=torch.float32).reshape(
        batch, n_patches, dim
    )
    hidden = torch.cat([cls, reg, patches], dim=1)
    grid = patch_tokens_to_grid(
        hidden,
        height=grid_h * 16,
        width=grid_w * 16,
        patch_size=16,
        num_register_tokens=registers,
    )
    assert grid.shape == (batch, dim, grid_h, grid_w)
    assert torch.equal(grid[0, :, 0, 0], patches[0, 0])
    assert torch.equal(grid[0, :, -1, -1], patches[0, -1])


def test_embedding_l2_normalization() -> None:
    tokens = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    out = l2_normalize_embeddings(tokens)
    assert torch.allclose(out[0, 0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(out[0, 1], torch.zeros(2))


def test_deterministic_memory_sampling_matches_helper() -> None:
    a = deterministic_candidate_pool(20_000, 10_000, seed=42)
    b = deterministic_candidate_pool(20_000, 10_000, seed=42)
    c = deterministic_candidate_pool(20_000, 10_000, seed=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 10_000


def test_nearest_neighbor_scoring_k1() -> None:
    memory = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    dist = nearest_neighbor_l2(queries, memory)
    assert dist.shape == (2,)
    assert dist[0] == pytest.approx(1.0)
    assert dist[1] == pytest.approx(0.0)


def test_cpu_mock_encoder_anomaly_map_and_image_score() -> None:
    cfg = DINOv3AnomalyConfig(
        input_size=32,
        candidate_pool_size=50,
        max_memory_patches=8,
        seed=42,
    )
    encoder = FrozenDINOv3PatchEncoder(
        MockProcessor(),
        TinyPatchViT(patch_size=16, dim=8),
        input_size=32,
        patch_size=16,
        device=torch.device("cpu"),
    )
    scorer = DINOv3PatchScorer(encoder, config=cfg)
    images = [torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8) for _ in range(3)]
    scorer.fit(images, log=None)
    assert scorer.memory is not None
    assert scorer.grid_hw == (2, 2)
    score, anomaly_map = scorer.score_image(images[0], native_hw=(48, 48))
    assert anomaly_map.shape == (48, 48)
    assert np.isfinite(score)
    patch_map = nearest_neighbor_l2(scorer.extract_patches(images[0]), scorer.memory).reshape(2, 2)
    assert score == pytest.approx(float(patch_map.max()))
    up = upsample_anomaly_map(patch_map.astype(np.float32), (48, 48))
    assert up.shape == (48, 48)


def test_load_official_weights_errors_without_silent_fallback() -> None:
    with pytest.raises(DINOWeightsError) as exc:
        load_official_dinov3("facebook/dinov2-small")
    assert "dinov3-vits16" in str(exc.value)
    instructions = dinov3_access_instructions()
    assert OFFICIAL_DINOV3_VITS16 in instructions
    assert "huggingface-cli login" in instructions
    assert "DINOv2" in instructions
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(DINOWeightsError) as missing:
            load_official_dinov3()
        assert "transformers" in str(missing.value).lower() or "huggingface-cli" in str(
            missing.value
        )
