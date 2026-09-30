"""Unit tests for MVTecDataset using tiny in-memory folder trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspectai.data import (
    IdentityTransform,
    MVTecDataset,
    MVTecLayoutError,
    MVTecPairingError,
    MVTecSample,
)
from inspectai.data import mvtec as mvtec_mod


def _save_rgb(path: Path, size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _save_mask(path: Path, size: tuple[int, int] = (8, 8), value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def _build_category(
    tmp_path: Path,
    category: str = "widget",
    *,
    train_n: int = 3,
    test_good_n: int = 2,
    defects: dict[str, int] | None = None,
    include_gt: bool = True,
    image_size: tuple[int, int] = (8, 8),
) -> Path:
    """Create a fake MVTec category under ``tmp_path / raw`` and return that raw root."""
    if defects is None:
        defects = {"scratch": 2, "dent": 1}
    root = tmp_path / "raw"
    cat = root / category
    (cat / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(train_n):
        _save_rgb(cat / "train" / "good" / f"{i:03d}.png", size=image_size)
    for i in range(test_good_n):
        _save_rgb(cat / "test" / "good" / f"{i:03d}.png", size=image_size)
    for defect, count in defects.items():
        for i in range(count):
            _save_rgb(cat / "test" / defect / f"{i:03d}.png", size=image_size)
            if include_gt:
                _save_mask(cat / "ground_truth" / defect / f"{i:03d}_mask.png", size=image_size)
    return root


def test_train_split_counts_labels_and_zero_masks(tmp_path: Path) -> None:
    root = _build_category(tmp_path, train_n=4)
    dataset = MVTecDataset(root, "widget", "train")
    assert len(dataset) == 4
    assert dataset.defect_types == ()
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, MVTecSample)
        assert sample.label == 0
        assert sample.defect_type == "good"
        assert sample.image.shape == (8, 8, 3)
        assert sample.image.dtype == np.uint8
        assert sample.mask.shape == (8, 8)
        assert sample.mask.dtype == np.uint8
        assert set(np.unique(sample.mask).tolist()).issubset({0})
        assert sample.image_path.is_file()
        assert sample.image_path.name.endswith(".png")


def test_test_split_discovers_defects_and_labels(tmp_path: Path) -> None:
    root = _build_category(tmp_path, test_good_n=2, defects={"scratch": 2, "dent": 1})
    dataset = MVTecDataset(root, "widget", "test")
    assert len(dataset) == 5
    assert dataset.defect_types == ("dent", "scratch")
    labels = [dataset[i].label for i in range(len(dataset))]
    types = [dataset[i].defect_type for i in range(len(dataset))]
    assert labels.count(0) == 2
    assert labels.count(1) == 3
    assert types.count("good") == 2
    assert types.count("scratch") == 2
    assert types.count("dent") == 1


def test_anomalous_masks_are_binary_and_normals_are_zero(tmp_path: Path) -> None:
    root = _build_category(tmp_path)
    dataset = MVTecDataset(root, "widget", "test")
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.image.shape[:2] == sample.mask.shape
        assert set(np.unique(sample.mask).tolist()).issubset({0, 1})
        if sample.label == 0:
            assert int(sample.mask.sum()) == 0
        else:
            assert int(sample.mask.sum()) > 0


def test_native_resolution_is_preserved(tmp_path: Path) -> None:
    root = _build_category(tmp_path, image_size=(6, 10), train_n=1, test_good_n=1, defects={"scratch": 1})
    train = MVTecDataset(root, "widget", "train")
    test = MVTecDataset(root, "widget", "test")
    assert train[0].image.shape == (10, 6, 3)
    assert test[0].image.shape[0] == 10
    assert test[0].image.shape[1] == 6


def test_identity_transform_hook(tmp_path: Path) -> None:
    root = _build_category(tmp_path, train_n=1, test_good_n=0, defects={})
    # test/good required even if empty defects — add one good test via helper: defects {} still creates test/good
    dataset = MVTecDataset(root, "widget", "train", transform=IdentityTransform())
    sample = dataset[0]
    assert sample.image.shape == (8, 8, 3)


def test_train_does_not_require_ground_truth(tmp_path: Path) -> None:
    root = _build_category(tmp_path, include_gt=False, defects={})
    dataset = MVTecDataset(root, "widget", "train")
    assert len(dataset) == 3


def test_test_without_defects_does_not_require_ground_truth(tmp_path: Path) -> None:
    root = _build_category(tmp_path, defects={}, include_gt=False)
    dataset = MVTecDataset(root, "widget", "test")
    assert len(dataset) == 2
    assert dataset.defect_types == ()


def test_missing_category_raises(tmp_path: Path) -> None:
    with pytest.raises(MVTecLayoutError, match="Missing category"):
        MVTecDataset(tmp_path / "raw", "missing", "train")


def test_invalid_split_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path)
    with pytest.raises(MVTecLayoutError, match="Invalid split"):
        MVTecDataset(root, "widget", "val")  # type: ignore[arg-type]


def test_missing_train_good_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path)
    good = root / "widget" / "train" / "good"
    for path in good.iterdir():
        path.unlink()
    good.rmdir()
    with pytest.raises(MVTecLayoutError, match="train/good"):
        MVTecDataset(root, "widget", "train")


def test_empty_train_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, train_n=0, defects={})
    with pytest.raises(MVTecLayoutError, match="No training images"):
        MVTecDataset(root, "widget", "train")


def test_missing_test_good_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path)
    good = root / "widget" / "test" / "good"
    for path in good.iterdir():
        path.unlink()
    good.rmdir()
    with pytest.raises(MVTecLayoutError, match="test/good"):
        MVTecDataset(root, "widget", "test")


def test_unexpected_train_class_folder_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path)
    _save_rgb(root / "widget" / "train" / "scratch" / "000.png")
    with pytest.raises(MVTecLayoutError, match="Unexpected train class"):
        MVTecDataset(root, "widget", "train")


def test_missing_mask_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, defects={"scratch": 1})
    mask = next((root / "widget" / "ground_truth" / "scratch").glob("*"))
    mask.unlink()
    with pytest.raises(MVTecPairingError, match="Missing masks"):
        MVTecDataset(root, "widget", "test")


def test_unmatched_mask_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, defects={"scratch": 1})
    _save_mask(root / "widget" / "ground_truth" / "scratch" / "999_mask.png")
    with pytest.raises(MVTecPairingError, match="Unmatched masks"):
        MVTecDataset(root, "widget", "test")


def test_unmatched_gt_class_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, defects={"scratch": 1})
    _save_mask(root / "widget" / "ground_truth" / "other" / "000_mask.png")
    with pytest.raises(MVTecPairingError, match="Unmatched ground_truth class"):
        MVTecDataset(root, "widget", "test")


def test_test_defects_without_ground_truth_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, include_gt=False, defects={"scratch": 1})
    with pytest.raises(MVTecLayoutError, match="ground_truth"):
        MVTecDataset(root, "widget", "test")


def test_image_mask_size_mismatch_raises(tmp_path: Path) -> None:
    root = _build_category(tmp_path, defects={"scratch": 1}, test_good_n=0)
    mask = root / "widget" / "ground_truth" / "scratch" / "000_mask.png"
    _save_mask(mask, size=(4, 4))
    dataset = MVTecDataset(root, "widget", "test")
    with pytest.raises(MVTecPairingError, match="spatial size mismatch"):
        _ = dataset[0]


def test_duplicate_indexed_paths_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _build_category(tmp_path, train_n=2, defects={})
    original = mvtec_mod._list_images

    def duplicated(class_dir: Path) -> list[Path]:
        paths = original(class_dir)
        return paths + paths[:1] if paths else paths

    monkeypatch.setattr(mvtec_mod, "_list_images", duplicated)
    with pytest.raises(MVTecLayoutError, match="Duplicate indexed"):
        MVTecDataset(root, "widget", "train")
