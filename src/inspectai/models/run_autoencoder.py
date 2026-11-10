"""Evaluate a trained autoencoder on the official test split (no training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.evaluation.metrics import binary_auroc, image_auroc_per_group
from inspectai.evaluation.reporting import write_analysis_figures, write_scores_csv
from inspectai.models.autoencoder import (
    ConvAutoencoder,
    image_anomaly_score,
    reconstruction_error_map,
)
from inspectai.training.autoencoder import resolve_device
from inspectai.training.preprocess import image_to_float01, upsample_map


def _uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.permute(1, 2, 0).cpu().numpy()


def evaluate_autoencoder(
    root: Path,
    category: str,
    checkpoint: Path,
    output_dir: Path,
    *,
    input_size: int = 256,
    device: torch.device | None = None,
) -> dict:
    device = device or resolve_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    test = MVTecTorchDataset(MVTecDataset(root, category, "test"))

    image_scores: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    maps: list[np.ndarray] = []
    rgb_images: list[np.ndarray] = []
    gt_masks: list[np.ndarray] = []
    image_paths: list[str] = []
    pixel_scores: list[np.ndarray] = []
    pixel_masks: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(test)):
            item = test[i]
            native_hw = (int(item["image"].shape[1]), int(item["image"].shape[2]))
            x = image_to_float01(item["image"], input_size).unsqueeze(0).to(device)
            recon = model(x)
            err = reconstruction_error_map(x[0].cpu(), recon[0].cpu())
            native_map = upsample_map(err, native_hw).numpy()
            score = float(image_anomaly_score(err).cpu())
            image_scores.append(score)
            labels.append(int(item["label"]))
            groups.append(str(item["defect_type"]))
            maps.append(native_map)
            rgb_images.append(_uint8_hwc(item["image"]))
            gt_masks.append(item["mask"].cpu().numpy())
            image_paths.append(str(item["image_path"]))
            pixel_scores.append(native_map.ravel())
            pixel_masks.append(item["mask"].cpu().numpy().ravel())

    y = np.asarray(labels)
    s = np.asarray(image_scores)
    g = np.asarray(groups)
    metrics = {
        "category": category,
        "n_test": len(test),
        "checkpoint": str(checkpoint),
        "config": {
            "input_size": input_size,
            "loss": "mse",
            "image_score": "spatial_max",
        },
        "image_auroc_overall": binary_auroc(y, s),
        "image_auroc_per_defect": image_auroc_per_group(y, s, g),
        "pixel_auroc": binary_auroc(np.concatenate(pixel_masks), np.concatenate(pixel_scores)),
    }

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
