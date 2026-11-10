"""Evaluate the frozen patch-kNN baseline. Train memory from good images only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.evaluation.case_selection import (
    flatten_selection,
    select_analysis_indices,
)
from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group
from inspectai.models.patch_knn import (
    FrozenResNetPatchEncoder,
    PatchKNNConfig,
    PatchKNNScorer,
)
from inspectai.visualization.overlays import anomaly_overlay, save_rgb
from inspectai.visualization.panels import analysis_filename, save_four_panel


def _uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.permute(1, 2, 0).cpu().numpy()


def run_patch_knn(
    root: Path,
    category: str,
    output_dir: Path,
    *,
    config: PatchKNNConfig | None = None,
    max_overlays: int = 8,
) -> dict:
    """Fit on train/good, score the official test split, write metrics and overlays.

    Returns the metrics dict. Does not look at test labels while fitting.
    """
    cfg = config or PatchKNNConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_np = MVTecDataset(root, category, "train")
    test_np = MVTecDataset(root, category, "test")
    train = MVTecTorchDataset(train_np)
    test = MVTecTorchDataset(test_np)

    encoder = FrozenResNetPatchEncoder(layer_name=cfg.feature_layer)
    scorer = PatchKNNScorer(encoder, config=cfg)
    train_images = [train[i]["image"] for i in range(len(train))]
    scorer.fit(train_images)

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
    metrics = {
        "category": category,
        "n_train": len(train),
        "n_test": len(test),
        "n_memory_patches": int(scorer.memory.shape[0]) if scorer.memory is not None else 0,
        "config": {
            "backbone": cfg.backbone,
            "feature_layer": cfg.feature_layer,
            "input_size": cfg.input_size,
            "k_neighbors": cfg.k_neighbors,
            "max_memory_patches": cfg.max_memory_patches,
            "memory_sample_seed": cfg.memory_sample_seed,
            "image_score": cfg.image_score,
        },
        "image_auroc_overall": binary_auroc(y, s),
        "image_auroc_per_defect": image_auroc_per_group(y, s, g),
        "pixel_auroc": binary_auroc(np.concatenate(pixel_masks), np.concatenate(pixel_scores)),
    }

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-s)
    for saved, idx in enumerate(order):
        if saved >= max_overlays:
            break
        name = f"{saved:02d}_{g[idx]}_{int(y[idx])}.png"
        blended = anomaly_overlay(rgb_images[idx], maps[idx])
        save_rgb(overlay_dir / name, blended)

    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "defect_type", "label", "anomaly_score"],
        )
        writer.writeheader()
        for path, defect, label, score in zip(image_paths, groups, labels, image_scores, strict=True):
            writer.writerow(
                {
                    "image_path": path,
                    "defect_type": defect,
                    "label": int(label),
                    "anomaly_score": float(score),
                }
            )

    analysis_root = output_dir / "analysis"
    selected = select_analysis_indices(y, g, s)
    bucket_dirs = {
        "per_defect": analysis_root / "per_defect",
        "high_score_good": analysis_root / "high_score_good",
        "low_score_defect": analysis_root / "low_score_defect",
    }
    for bucket, tag, idx in flatten_selection(selected):
        fname = analysis_filename(
            defect_type=groups[idx],
            label=int(labels[idx]),
            score=float(image_scores[idx]),
            image_path=image_paths[idx],
        )
        title = (
            f"{tag} | {groups[idx]} | label={int(labels[idx])} | "
            f"score={image_scores[idx]:.4f} | {Path(image_paths[idx]).name}"
        )
        save_four_panel(
            bucket_dirs[bucket] / fname,
            rgb_images[idx],
            gt_masks[idx],
            maps[idx],
            title=title,
        )

    return metrics
