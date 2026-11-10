"""Shared score CSV and four-panel analysis writers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from inspectai.evaluation.case_selection import (
    flatten_selection,
    select_analysis_indices,
)
from inspectai.visualization.panels import analysis_filename, save_four_panel


def write_scores_csv(
    path: Path,
    image_paths: list[str],
    groups: list[str],
    labels: list[int],
    scores: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "defect_type", "label", "anomaly_score"],
        )
        writer.writeheader()
        for image_path, defect, label, score in zip(
            image_paths, groups, labels, scores, strict=True
        ):
            writer.writerow(
                {
                    "image_path": image_path,
                    "defect_type": defect,
                    "label": int(label),
                    "anomaly_score": float(score),
                }
            )


def write_analysis_figures(
    analysis_root: Path,
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    rgb_images: list[np.ndarray],
    gt_masks: list[np.ndarray],
    maps: list[np.ndarray],
    image_paths: list[str],
) -> None:
    selected = select_analysis_indices(labels, groups, scores)
    bucket_dirs = {
        "per_defect": analysis_root / "per_defect",
        "high_score_good": analysis_root / "high_score_good",
        "low_score_defect": analysis_root / "low_score_defect",
    }
    for bucket, tag, idx in flatten_selection(selected):
        fname = analysis_filename(
            defect_type=str(groups[idx]),
            label=int(labels[idx]),
            score=float(scores[idx]),
            image_path=image_paths[idx],
        )
        title = (
            f"{tag} | {groups[idx]} | label={int(labels[idx])} | "
            f"score={float(scores[idx]):.4f} | {Path(image_paths[idx]).name}"
        )
        save_four_panel(
            bucket_dirs[bucket] / fname,
            rgb_images[idx],
            gt_masks[idx],
            maps[idx],
            title=title,
        )
