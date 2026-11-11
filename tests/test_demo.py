"""Lightweight tests for Streamlit demo helpers. Does not launch Streamlit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inspectai.inference.demo import (
    BUILD_ARTIFACT_COMMAND,
    EVALUATION_SUMMARY,
    LOCKED_EVALUATION,
    anomaly_heatmap_figure,
    artifact_is_ready,
    format_inspection_metrics,
    friendly_model_name,
    missing_artifact_message,
    resolve_artifact_dir,
    user_facing_error,
)
from inspectai.models.dinov3_anomaly import DINOWeightsError


def test_locked_evaluation_copy_is_accurate() -> None:
    by_name = {row["name"]: row for row in LOCKED_EVALUATION}
    assert by_name["Autoencoder"] == {
        "name": "Autoencoder",
        "image_auroc": "0.6158",
        "pixel_auroc": "0.5898",
    }
    assert by_name["ResNet18 patch-kNN"]["image_auroc"] == "0.9800"
    assert by_name["DINOv3"]["pixel_auroc"] == "0.9663"
    assert "highest image-level AUROC" in EVALUATION_SUMMARY
    assert "strongest localization" in EVALUATION_SUMMARY


def test_missing_artifact_message_includes_build_command(tmp_path: Path) -> None:
    missing = tmp_path / "empty"
    assert not artifact_is_ready(missing)
    message = missing_artifact_message(missing)
    assert BUILD_ARTIFACT_COMMAND in message
    assert str(missing) in message


def test_artifact_ready_requires_both_files(tmp_path: Path) -> None:
    (tmp_path / "memory.npz").write_bytes(b"not-a-real-npz")
    assert not artifact_is_ready(tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert artifact_is_ready(tmp_path)


def test_metric_formatting_and_model_name() -> None:
    metrics = format_inspection_metrics(
        anomaly_score=0.5134,
        latency_ms=1628.2,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        input_hw=(1024, 1024),
        patch_grid=(32, 32),
    )
    assert metrics["Anomaly score"] == "0.5134"
    assert metrics["Inference"] == "1628 ms"
    assert metrics["Model"] == "DINOv3 ViT-S/16"
    assert metrics["Input"] == "1024 × 1024"
    assert metrics["Patch grid"] == "32 × 32"
    assert friendly_model_name("other-backbone") == "other-backbone"


def test_heatmap_figure_uses_colorbar() -> None:
    heat = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    fig = anomaly_heatmap_figure(heat)
    assert len(fig.axes) >= 2
    fig.clf()


def test_user_facing_error_surfaces_dino_and_artifact_messages(tmp_path: Path) -> None:
    dino = DINOWeightsError("accept the DINOv3 license")
    assert "DINOv3 license" in user_facing_error(dino)
    assert BUILD_ARTIFACT_COMMAND in user_facing_error(FileNotFoundError("x"), artifact_dir=tmp_path)


def test_resolve_artifact_dir_from_repo_root(tmp_path: Path) -> None:
    resolved = resolve_artifact_dir(repo_root=tmp_path)
    assert resolved == tmp_path / "outputs" / "artifacts" / "dinov3_transistor"
