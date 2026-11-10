"""Evaluate frozen DINOv3 patch kNN. Fit on train/good only; official test last."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group
from inspectai.evaluation.reporting import write_analysis_figures, write_scores_csv
from inspectai.models.dinov3_anomaly import (
    DINOv3AnomalyConfig,
    DINOv3PatchScorer,
    FrozenDINOv3PatchEncoder,
    load_official_dinov3,
)


def _uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.permute(1, 2, 0).cpu().numpy()


def run_dinov3(
    root: Path,
    category: str,
    output_dir: Path,
    *,
    config: DINOv3AnomalyConfig | None = None,
) -> dict:
    """Fit memory on train/good, score official test, write metrics and analysis."""
    cfg = config or DINOv3AnomalyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_official_dinov3(cfg.model_id)
    encoder = FrozenDINOv3PatchEncoder(
        processor,
        model,
        input_size=cfg.input_size,
        patch_size=cfg.patch_size,
        num_register_tokens=cfg.num_register_tokens,
    )
    train = MVTecTorchDataset(MVTecDataset(root, category, "train"))
    test = MVTecTorchDataset(MVTecDataset(root, category, "test"))
    scorer = DINOv3PatchScorer(encoder, config=cfg)
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

    print("starting official test scoring")
    n_test = len(test)
    for i in range(n_test):
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
        done = i + 1
        if done % 10 == 0 or done == n_test:
            print(f"test progress: {done}/{n_test}")

    y = np.asarray(labels)
    s = np.asarray(image_scores)
    g = np.asarray(groups)
    meta = scorer.memory_report()
    metrics = {
        "category": category,
        "n_train": len(train),
        "n_test": len(test),
        "config": {
            "backbone": cfg.model_id,
            "input_size": cfg.input_size,
            "patch_size": cfg.patch_size,
            "num_register_tokens": cfg.num_register_tokens,
            "k": cfg.k,
            "candidate_pool_size": cfg.candidate_pool_size,
            "max_memory_patches": cfg.max_memory_patches,
            "seed": cfg.seed,
            "image_score": "spatial_max",
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
