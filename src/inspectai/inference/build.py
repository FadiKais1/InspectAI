"""Fit the locked M7 DINOv3 memory from train/good only and write an artifact."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.inference.artifact import save_memory_artifact
from inspectai.inference.inspector import resolve_inspector_device
from inspectai.models.dinov3_anomaly import (
    DINOv3AnomalyConfig,
    DINOv3PatchScorer,
    FrozenDINOv3PatchEncoder,
    load_official_dinov3,
)


def build_dinov3_memory_artifact(
    root: Path,
    category: str,
    output_dir: Path,
    *,
    device: str | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Extract train/good features with locked M7 config. Never reads the test split."""
    cfg = DINOv3AnomalyConfig()
    train = MVTecTorchDataset(MVTecDataset(root, category, "train"))
    processor, model = load_official_dinov3(cfg.model_id)
    encoder = FrozenDINOv3PatchEncoder(
        processor,
        model,
        input_size=cfg.input_size,
        patch_size=cfg.patch_size,
        num_register_tokens=cfg.num_register_tokens,
        device=resolve_inspector_device(device),
    )
    scorer = DINOv3PatchScorer(encoder, config=cfg)
    scorer.fit([train[i]["image"] for i in range(len(train))], log=log)
    if scorer.memory is None:
        raise RuntimeError("memory fit produced no embeddings")
    meta = scorer.memory_report()
    meta.update(
        {
            "category": category,
            "split": "train",
            "train_subset": "good",
            "n_train": len(train),
            "image_score": "spatial_max",
            "num_register_tokens": cfg.num_register_tokens,
        }
    )
    save_memory_artifact(output_dir, scorer.memory, meta)
    if log is not None:
        log(f"wrote {output_dir / 'memory.npz'}")
        log(f"wrote {output_dir / 'metadata.json'}")
    return meta
