"""Save/load a fitted DINOv3 normal-memory artifact (npz + JSON, no pickle)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspectai.models.dinov3_anomaly import OFFICIAL_DINOV3_VITS16, DINOv3AnomalyConfig

MEMORY_FILENAME = "memory.npz"
METADATA_FILENAME = "metadata.json"

LOCKED_M7_FIELDS = {
    "backbone": OFFICIAL_DINOV3_VITS16,
    "input_size": 512,
    "patch_size": 16,
    "num_register_tokens": 4,
    "embedding_dim": 384,
    "k": 1,
    "seed": 42,
    "candidate_pool_size": 50_000,
    "max_memory_patches": 10_000,
    "image_score": "spatial_max",
    "selection": "deterministic_random",
}


class ArtifactError(ValueError):
    """Memory artifact is missing, inconsistent, or not the locked M7 config."""


def locked_m7_metadata(**extra: object) -> dict:
    cfg = DINOv3AnomalyConfig()
    meta = {
        "backbone": cfg.model_id,
        "input_size": cfg.input_size,
        "patch_size": cfg.patch_size,
        "num_register_tokens": cfg.num_register_tokens,
        "embedding_dim": LOCKED_M7_FIELDS["embedding_dim"],
        "k": cfg.k,
        "seed": cfg.seed,
        "candidate_pool_size": cfg.candidate_pool_size,
        "max_memory_patches": cfg.max_memory_patches,
        "image_score": "spatial_max",
        "selection": "deterministic_random",
        "grid_hw": [cfg.input_size // cfg.patch_size, cfg.input_size // cfg.patch_size],
        "split": "train",
        "train_subset": "good",
    }
    meta.update(extra)
    return meta


def validate_artifact_consistency(metadata: dict, embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2:
        raise ArtifactError(f"embeddings must be (N, D), got {embeddings.shape}")
    n_memory, dim = embeddings.shape
    reported_n = metadata.get("n_memory_patches")
    reported_d = metadata.get("embedding_dim")
    if reported_n is not None and int(reported_n) != int(n_memory):
        raise ArtifactError(
            f"n_memory_patches={reported_n} does not match embeddings {n_memory}"
        )
    if reported_d is not None and int(reported_d) != int(dim):
        raise ArtifactError(f"embedding_dim={reported_d} does not match embeddings D={dim}")


def validate_locked_m7_metadata(metadata: dict) -> None:
    for key, expected in LOCKED_M7_FIELDS.items():
        if key not in metadata:
            raise ArtifactError(f"artifact metadata missing {key!r}")
        got = metadata[key]
        if got != expected:
            raise ArtifactError(
                f"artifact {key}={got!r} does not match locked M7 value {expected!r}"
            )
    grid = metadata.get("grid_hw")
    expected_grid = [
        LOCKED_M7_FIELDS["input_size"] // LOCKED_M7_FIELDS["patch_size"],
        LOCKED_M7_FIELDS["input_size"] // LOCKED_M7_FIELDS["patch_size"],
    ]
    if grid is not None and list(grid) != expected_grid:
        raise ArtifactError(f"grid_hw={grid!r} does not match locked {expected_grid}")


def save_memory_artifact(
    output_dir: Path,
    embeddings: np.ndarray,
    metadata: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = np.asarray(embeddings, dtype=np.float32)
    payload = dict(metadata)
    payload["n_memory_patches"] = int(bank.shape[0])
    payload["embedding_dim"] = int(bank.shape[1])
    validate_artifact_consistency(payload, bank)
    np.savez_compressed(output_dir / MEMORY_FILENAME, embeddings=bank)
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return output_dir


def load_memory_artifact(artifact_dir: Path) -> tuple[np.ndarray, dict]:
    artifact_dir = Path(artifact_dir)
    memory_path = artifact_dir / MEMORY_FILENAME
    meta_path = artifact_dir / METADATA_FILENAME
    if not memory_path.is_file():
        raise ArtifactError(f"missing {memory_path}")
    if not meta_path.is_file():
        raise ArtifactError(f"missing {meta_path}")
    with np.load(memory_path) as data:
        if "embeddings" not in data:
            raise ArtifactError("memory.npz must contain array 'embeddings'")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ArtifactError("metadata.json must be an object")
    validate_artifact_consistency(metadata, embeddings)
    return embeddings, metadata
