"""Four-panel qualitative figures for anomaly-detection analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inspectai.visualization.overlays import anomaly_overlay


def save_four_panel(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    score_map: np.ndarray,
    *,
    title: str,
) -> None:
    """Save RGB | GT mask | raw heatmap | overlay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = anomaly_overlay(image, score_map)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT mask")
    heat = axes[2].imshow(score_map, cmap="inferno")
    axes[2].set_title("Anomaly heatmap")
    fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def analysis_filename(
    *,
    defect_type: str,
    label: int,
    score: float,
    image_path: str,
) -> str:
    stem = Path(image_path).stem
    safe_defect = defect_type.replace("/", "_")
    return f"{safe_defect}_label{int(label)}_score{score:.4f}_{stem}.png"
