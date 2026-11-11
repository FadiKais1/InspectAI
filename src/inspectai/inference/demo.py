"""Non-UI helpers for the Streamlit demo. Does not change Inspector scoring."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from inspectai.models.dinov3_anomaly import DINOWeightsError

DEFAULT_ARTIFACT_DIR = Path("outputs/artifacts/dinov3_transistor")
BUILD_ARTIFACT_COMMAND = (
    "python scripts/build_dinov3_memory.py --root data/raw "
    "--category transistor --output outputs/artifacts/dinov3_transistor"
)
STREAMLIT_LAUNCH_COMMAND = "streamlit run app/streamlit_app.py"

LOCKED_EVALUATION = (
    {
        "name": "Autoencoder",
        "image_auroc": "0.6158",
        "pixel_auroc": "0.5898",
    },
    {
        "name": "ResNet18 patch-kNN",
        "image_auroc": "0.9800",
        "pixel_auroc": "0.7866",
    },
    {
        "name": "PatchCore-style",
        "image_auroc": "0.9479",
        "pixel_auroc": "0.7689",
    },
    {
        "name": "DINOv3",
        "image_auroc": "0.9646",
        "pixel_auroc": "0.9663",
    },
)

EVALUATION_SUMMARY = (
    "DINOv3 provided the strongest localization performance and the best "
    "overall detection/localization balance, while the simpler ResNet "
    "patch-kNN achieved the highest image-level AUROC."
)


def resolve_artifact_dir(explicit: str | Path | None = None, *, repo_root: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if repo_root is None:
        return DEFAULT_ARTIFACT_DIR
    return repo_root / DEFAULT_ARTIFACT_DIR


def artifact_is_ready(artifact_dir: Path) -> bool:
    return (artifact_dir / "memory.npz").is_file() and (artifact_dir / "metadata.json").is_file()


def missing_artifact_message(artifact_dir: Path | None = None) -> str:
    location = artifact_dir if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
    return (
        f"DINOv3 memory artifact not found at `{location}`.\n\n"
        "The demo uses a saved normal memory bank and does not rebuild it. "
        "Build the artifact with:\n\n"
        f"`{BUILD_ARTIFACT_COMMAND}`"
    )


def friendly_model_name(model_name: str) -> str:
    text = model_name.lower()
    if "dinov3" in text and "vits16" in text.replace("-", "").replace("_", ""):
        return "DINOv3 ViT-S/16"
    if "dinov3" in text:
        return "DINOv3 ViT-S/16"
    return model_name


def format_hw(hw: tuple[int, int]) -> str:
    return f"{int(hw[0])} × {int(hw[1])}"


def format_inspection_metrics(
    *,
    anomaly_score: float,
    latency_ms: float,
    model_name: str,
    input_hw: tuple[int, int],
    patch_grid: tuple[int, int],
) -> dict[str, str]:
    return {
        "Anomaly score": f"{anomaly_score:.4f}",
        "Inference": f"{latency_ms:.0f} ms",
        "Model": friendly_model_name(model_name),
        "Input": format_hw(input_hw),
        "Patch grid": format_hw(patch_grid),
    }


def anomaly_heatmap_figure(anomaly_map: np.ndarray) -> Figure:
    """Readable heatmap with colorbar. Does not change overlay or scoring."""
    heat = np.asarray(anomaly_map, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
    image = ax.imshow(heat, cmap="inferno")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Anomaly heatmap")
    return fig


def user_facing_error(exc: BaseException, *, artifact_dir: Path | None = None) -> str:
    if isinstance(exc, FileNotFoundError):
        return missing_artifact_message(artifact_dir)
    if isinstance(exc, DINOWeightsError):
        return str(exc)
    text = str(exc).strip() or exc.__class__.__name__
    if "artifact" in text.lower() or "memory.npz" in text.lower():
        return missing_artifact_message(artifact_dir)
    return text
