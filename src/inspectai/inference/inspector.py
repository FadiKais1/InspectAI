"""Reusable inference wrapper around the locked M7 DINOv3 patch memory."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL.Image import Image as PILImage

from inspectai.inference.artifact import (
    ArtifactError,
    load_memory_artifact,
    validate_locked_m7_metadata,
)
from inspectai.inference.images import to_uint8_rgb
from inspectai.inference.types import InspectionResult
from inspectai.models.dinov3_anomaly import (
    DINOv3AnomalyConfig,
    DINOv3PatchScorer,
    FrozenDINOv3PatchEncoder,
    load_official_dinov3,
)
from inspectai.visualization.overlays import anomaly_overlay


def resolve_inspector_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def config_from_metadata(metadata: dict) -> DINOv3AnomalyConfig:
    return DINOv3AnomalyConfig(
        input_size=int(metadata["input_size"]),
        candidate_pool_size=int(metadata["candidate_pool_size"]),
        max_memory_patches=int(metadata["max_memory_patches"]),
        seed=int(metadata["seed"]),
        k=int(metadata["k"]),
        model_id=str(metadata["backbone"]),
        patch_size=int(metadata["patch_size"]),
        num_register_tokens=int(metadata["num_register_tokens"]),
    )


class Inspector:
    """Frozen DINOv3 patch-memory inspector. Does not rebuild train memory."""

    def __init__(
        self,
        scorer: DINOv3PatchScorer,
        metadata: dict,
        *,
        model_name: str | None = None,
    ) -> None:
        if scorer.memory is None:
            raise RuntimeError("scorer must have a fitted memory bank")
        self.scorer = scorer
        self.metadata = dict(metadata)
        self.model_name = model_name or str(metadata.get("backbone", "dinov3"))

    @classmethod
    def from_fitted(
        cls,
        encoder: FrozenDINOv3PatchEncoder,
        embeddings: np.ndarray,
        metadata: dict,
        *,
        require_locked_config: bool = False,
    ) -> Inspector:
        if require_locked_config:
            validate_locked_m7_metadata(metadata)
        cfg = config_from_metadata(metadata)
        scorer = DINOv3PatchScorer(encoder, config=cfg)
        grid = metadata.get("grid_hw") or [
            cfg.input_size // cfg.patch_size,
            cfg.input_size // cfg.patch_size,
        ]
        scorer.load_fitted_memory(
            embeddings,
            grid_hw=(int(grid[0]), int(grid[1])),
            n_candidate_patches=metadata.get("n_candidate_patches"),
            n_pool_patches=metadata.get("n_pool_patches"),
        )
        return cls(scorer, metadata, model_name=str(metadata.get("backbone")))

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> Inspector:
        embeddings, metadata = load_memory_artifact(Path(artifact_dir))
        validate_locked_m7_metadata(metadata)
        resolved = resolve_inspector_device(device)
        processor, model = load_official_dinov3(str(metadata["backbone"]))
        encoder = FrozenDINOv3PatchEncoder(
            processor,
            model,
            input_size=int(metadata["input_size"]),
            patch_size=int(metadata["patch_size"]),
            num_register_tokens=int(metadata["num_register_tokens"]),
            device=resolved,
        )
        return cls.from_fitted(
            encoder,
            embeddings,
            metadata,
            require_locked_config=True,
        )

    def inspect(
        self,
        image: np.ndarray | PILImage,
        *,
        threshold: float | None = None,
    ) -> InspectionResult:
        rgb = np.array(to_uint8_rgb(image), copy=True)
        chw = torch.from_numpy(rgb).permute(2, 0, 1)
        native_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        started = time.perf_counter()
        score, anomaly_map = self.scorer.score_image(chw, native_hw)
        overlay = anomaly_overlay(rgb, anomaly_map)
        latency_ms = (time.perf_counter() - started) * 1000.0
        grid = self.scorer.grid_hw
        if grid is None:
            raise ArtifactError("scorer is missing patch grid metadata")
        return InspectionResult(
            anomaly_score=float(score),
            anomaly_map=np.asarray(anomaly_map),
            overlay=overlay,
            latency_ms=float(latency_ms),
            model_name=self.model_name,
            input_hw=native_hw,
            patch_grid=(int(grid[0]), int(grid[1])),
            threshold=threshold,
        )
