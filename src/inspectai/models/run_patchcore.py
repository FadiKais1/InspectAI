"""Evaluate the PatchCore-style scorer. Fit on train/good only; test at the end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group
from inspectai.evaluation.reporting import write_analysis_figures, write_scores_csv
from inspectai.models.patchcore import (
    FrozenResNetMultiScaleEncoder,
    PatchCoreConfig,
    PatchCoreScorer,
)


def _uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.permute(1, 2, 0).cpu().numpy()


def run_patchcore(
    root: Path,
    category: str,
    output_dir: Path,
    *,
    config: PatchCoreConfig | None = None,
) -> dict:
    """Fit coreset on train/good, score official test, write metrics and analysis."""
    cfg = config or PatchCoreConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = MVTecTorchDataset(MVTecDataset(root, category, "train"))
    test = MVTecTorchDataset(MVTecDataset(root, category, "test"))
    encoder = FrozenResNetMultiScaleEncoder()
    scorer = PatchCoreScorer(encoder, config=cfg)
    scorer.fit([train[i]["image"] for i in range(len(train))])

    image_scores: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    maps: list[np.ndarray] = []
    rgb_images: list[np.ndarray] = []
    gt_masks: list[np.ndarray] = []
    image_paths: list[str] = []
    pixel_scores: list[np.ndarray] = []
    pixel_masks: list[np.ndarray] = []

    for i in range(len(test)):
        item = test[i]
        native_hw = (int(item["image"].shape[1]), int(item["image"].shape[2]))
        score, score_map = scorer.score_image(item["image"], native_hw)
        image_scores.append(score)
        labels.append(int(item["label"]))
        groups.append(str(item["defect_type"]))
        maps.append(score_map)
        rgb_images.append(_uint8_hwc(item["image"]))
        gt_masks.append(item["mask"].cpu().numpy())
        image_paths.append(str(item["image_path"]))
        pixel_scores.append(score_map.ravel())
        pixel_masks.append(item["mask"].cpu().numpy().ravel())

    y = np.asarray(labels)
    s = np.asarray(image_scores)
    g = np.asarray(groups)
    meta = scorer.memory_metadata()
    metrics = {
        "category": category,
        "n_train": len(train),
        "n_test": len(test),
        "config": {
            "backbone": cfg.backbone,
            "feature_layers": list(cfg.feature_layers),
            "input_size": cfg.input_size,
            "k_neighbors": cfg.k_neighbors,
            "max_memory_patches": cfg.max_memory_patches,
            "seed": cfg.seed,
            "image_score": cfg.image_score,
            "projection_dim": cfg.projection_dim,
        },
        "memory": meta,
        "image_auroc_overall": binary_auroc(y, s),
        "image_auroc_per_defect": image_auroc_per_group(y, s, g),
        "pixel_auroc": binary_auroc(np.concatenate(pixel_masks), np.concatenate(pixel_scores)),
    }

    (output_dir / "memory.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_scores_csv(output_dir / "scores.csv", image_paths, groups, labels, image_scores)
    write_analysis_figures(
        output_dir / "analysis",
        labels=y,
        groups=g,
        scores=s,
        rgb_images=rgb_images,
        gt_masks=gt_masks,
        maps=maps,
        image_paths=image_paths,
    )
    return metrics
