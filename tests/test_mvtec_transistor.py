"""Integration tests against local MVTec AD transistor data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspectai.data import MVTecDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSISTOR_ROOT = REPO_ROOT / "data" / "raw"
TRANSISTOR_DIR = TRANSISTOR_ROOT / "transistor"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TRANSISTOR_DIR.is_dir(),
        reason="local data/raw/transistor is not available",
    ),
]


def test_transistor_train_length_and_labels() -> None:
    dataset = MVTecDataset(TRANSISTOR_ROOT, "transistor", "train")
    assert len(dataset) == 213
    assert dataset.defect_types == ()
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.label == 0
        assert sample.defect_type == "good"
        assert sample.image.shape == (1024, 1024, 3)
        assert sample.mask.shape == (1024, 1024)
        assert sample.mask.dtype == np.uint8
        assert int(sample.mask.sum()) == 0
        assert sample.image_path.is_file()


def test_transistor_test_counts_and_dynamic_defects() -> None:
    dataset = MVTecDataset(TRANSISTOR_ROOT, "transistor", "test")
    assert len(dataset) == 100
    discovered = set(dataset.defect_types)
    on_disk = {
        path.name
        for path in (TRANSISTOR_DIR / "test").iterdir()
        if path.is_dir() and path.name != "good"
    }
    assert discovered == on_disk
    assert len(discovered) == 4

    normal = 0
    anomalous = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.image.shape[:2] == sample.mask.shape
        assert sample.image.shape == (1024, 1024, 3)
        if sample.label == 0:
            normal += 1
            assert sample.defect_type == "good"
            assert int(sample.mask.sum()) == 0
        else:
            anomalous += 1
            assert sample.defect_type in discovered
            assert int(sample.mask.sum()) > 0
            assert set(np.unique(sample.mask).tolist()).issubset({0, 1})
    assert normal == 60
    assert anomalous == 40
