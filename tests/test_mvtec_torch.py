"""Smoke tests for the optional PyTorch MVTec adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from inspectai.data import MVTecDataset, MVTecTorchDataset


def _save_rgb(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _save_mask(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _build_category(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    cat = root / "widget"
    (cat / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_rgb(cat / "train" / "good" / f"{i:03d}.png")
    for i in range(2):
        _save_rgb(cat / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _save_rgb(cat / "test" / "scratch" / f"{i:03d}.png")
        _save_mask(cat / "ground_truth" / "scratch" / f"{i:03d}_mask.png")
    return root


def test_torch_sample_is_chw_uint8_native_size(tmp_path: Path) -> None:
    base = MVTecDataset(_build_category(tmp_path), "widget", "train")
    dataset = MVTecTorchDataset(base)
    assert len(dataset) == 4
    assert dataset.defect_types == ()

    item = dataset[0]
    assert item["image"].shape == (3, 8, 8)
    assert item["image"].dtype == torch.uint8
    assert item["mask"].shape == (8, 8)
    assert item["mask"].dtype == torch.uint8
    assert int(item["mask"].sum()) == 0
    assert item["label"] == 0
    assert item["defect_type"] == "good"
    assert Path(item["image_path"]).is_file()


def test_torch_dataloader_batches_without_resize(tmp_path: Path) -> None:
    base = MVTecDataset(_build_category(tmp_path), "widget", "test")
    dataset = MVTecTorchDataset(base)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["mask"].shape == (2, 8, 8)
    assert batch["image"].dtype == torch.uint8
    assert len(batch["defect_type"]) == 2
    assert len(batch["image_path"]) == 2
    assert dataset.defect_types == ("scratch",)
