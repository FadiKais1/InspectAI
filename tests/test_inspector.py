"""Lightweight Inspector / artifact tests. Does not download DINOv3 or run MVTec."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from inspectai.inference.artifact import (
    ArtifactError,
    load_memory_artifact,
    locked_m7_metadata,
    save_memory_artifact,
    validate_locked_m7_metadata,
)
from inspectai.inference.images import to_uint8_rgb
from inspectai.inference.inspector import Inspector
from inspectai.inference.types import InspectionResult
from inspectai.models.dinov3_anomaly import (
    OFFICIAL_DINOV3_VITS16,
    FrozenDINOv3PatchEncoder,
)


class MockProcessor:
    def __call__(self, images, return_tensors="pt", do_resize=False):
        arr = np.asarray(images, dtype=np.float32) / 255.0
        pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": pixel}


class TinyPatchViT(nn.Module):
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
        zeros_cls = torch.zeros(batch, 1, dim, device=pixel_values.device, dtype=pixel_values.dtype)
        zeros_reg = torch.zeros(
            batch,
            self.config.num_register_tokens,
            dim,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        return SimpleNamespace(last_hidden_state=torch.cat([zeros_cls, zeros_reg, flat], dim=1))


def _tiny_encoder(input_size: int = 32, dim: int = 8) -> FrozenDINOv3PatchEncoder:
    return FrozenDINOv3PatchEncoder(
        MockProcessor(),
        TinyPatchViT(patch_size=16, dim=dim),
        input_size=input_size,
        patch_size=16,
        device=torch.device("cpu"),
    )


def _tiny_metadata(*, dim: int = 8, n_memory: int = 6) -> dict:
    return {
        "backbone": OFFICIAL_DINOV3_VITS16,
        "input_size": 32,
        "patch_size": 16,
        "num_register_tokens": 4,
        "embedding_dim": dim,
        "k": 1,
        "seed": 42,
        "candidate_pool_size": 50,
        "max_memory_patches": n_memory,
        "image_score": "spatial_max",
        "selection": "deterministic_random",
        "grid_hw": [2, 2],
        "n_memory_patches": n_memory,
        "n_candidate_patches": 12,
        "n_pool_patches": 8,
    }


def test_artifact_save_load_round_trip(tmp_path) -> None:
    embeddings = np.linspace(0.0, 1.0, 20 * 8, dtype=np.float32).reshape(20, 8)
    metadata = {
        "backbone": OFFICIAL_DINOV3_VITS16,
        "embedding_dim": 8,
        "n_memory_patches": 20,
        "seed": 42,
    }
    save_memory_artifact(tmp_path, embeddings, metadata)
    loaded, meta = load_memory_artifact(tmp_path)
    assert loaded.shape == (20, 8)
    assert np.allclose(loaded, embeddings)
    assert meta["seed"] == 42
    assert meta["n_memory_patches"] == 20


def test_metadata_mismatch_raises() -> None:
    meta = locked_m7_metadata()
    validate_locked_m7_metadata(meta)
    bad = dict(meta)
    bad["backbone"] = "facebook/dinov2-small"
    with pytest.raises(ArtifactError, match="locked M7"):
        validate_locked_m7_metadata(bad)
    bad_size = dict(meta)
    bad_size["input_size"] = 224
    with pytest.raises(ArtifactError, match="input_size"):
        validate_locked_m7_metadata(bad_size)


def test_inspector_load_rejects_mismatched_artifact_before_weights(tmp_path) -> None:
    embeddings = np.zeros((10, 384), dtype=np.float32)
    meta = locked_m7_metadata(n_memory_patches=10, embedding_dim=384)
    meta["backbone"] = "not-the-locked-model"
    save_memory_artifact(tmp_path, embeddings, meta)
    with pytest.raises(ArtifactError, match="backbone"):
        Inspector.load(tmp_path, device="cpu")


def test_inspection_result_structure_and_score_type() -> None:
    encoder = _tiny_encoder()
    memory = np.zeros((6, 8), dtype=np.float32)
    memory[:, 0] = 1.0
    inspector = Inspector.from_fitted(encoder, memory, _tiny_metadata())
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[10:20, 10:20] = 255
    result = inspector.inspect(rgb)
    assert isinstance(result, InspectionResult)
    assert isinstance(result.anomaly_score, float)
    assert np.isfinite(result.anomaly_score)
    assert result.anomaly_map.shape == (48, 48)
    assert result.overlay.shape == (48, 48, 3)
    assert result.overlay.dtype == np.uint8
    assert result.input_hw == (48, 48)
    assert result.patch_grid == (2, 2)
    assert result.model_name == OFFICIAL_DINOV3_VITS16
    assert result.threshold is None
    assert "above_threshold" not in result.summary()
    assert result.latency_ms >= 0.0


def test_inspect_accepts_pil_and_numpy() -> None:
    encoder = _tiny_encoder()
    memory = np.eye(8, dtype=np.float32)[:6]
    inspector = Inspector.from_fitted(encoder, memory, _tiny_metadata())
    array = np.random.default_rng(0).integers(0, 256, size=(40, 36, 3), dtype=np.uint8)
    pil = Image.fromarray(array)
    from_array = inspector.inspect(array)
    from_pil = inspector.inspect(pil)
    assert from_array.anomaly_map.shape == (40, 36)
    assert from_pil.anomaly_map.shape == (40, 36)
    assert from_array.anomaly_score == pytest.approx(from_pil.anomaly_score)


def test_uint8_rgb_conversion() -> None:
    pil = Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8))
    assert to_uint8_rgb(pil).shape == (4, 5, 3)
    floats = np.ones((4, 5, 3), dtype=np.float32)
    converted = to_uint8_rgb(floats)
    assert converted.dtype == np.uint8
    assert converted.max() == 255
    with pytest.raises(ValueError):
        to_uint8_rgb(np.zeros((4, 5), dtype=np.uint8))


def test_deterministic_mock_inference() -> None:
    encoder = _tiny_encoder()
    memory = np.zeros((6, 8), dtype=np.float32)
    memory[0, 0] = 1.0
    inspector = Inspector.from_fitted(encoder, memory, _tiny_metadata())
    image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8).permute(1, 2, 0).numpy()
    a = inspector.inspect(image)
    b = inspector.inspect(image)
    assert a.anomaly_score == pytest.approx(b.anomaly_score)
    assert np.allclose(a.anomaly_map, b.anomaly_map)
