"""Thin PyTorch adapter over :class:`MVTecDataset`.

Converts native-resolution NumPy samples to tensors. Does not resize,
normalize, or train a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.types import MVTecSample


def sample_to_tensors(sample: MVTecSample) -> dict[str, Any]:
    """Map an ``MVTecSample`` to a DataLoader-friendly dict of tensors."""
    image = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).contiguous()
    mask = torch.from_numpy(sample.mask.copy())
    return {
        "image": image,
        "mask": mask,
        "label": int(sample.label),
        "defect_type": sample.defect_type,
        "image_path": str(sample.image_path),
    }


class MVTecTorchDataset(Dataset[Mapping[str, Any]]):
    """``torch.utils.data.Dataset`` wrapper around a NumPy ``MVTecDataset``."""

    def __init__(self, dataset: MVTecDataset) -> None:
        self.dataset = dataset

    @property
    def defect_types(self) -> tuple[str, ...]:
        return self.dataset.defect_types

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return sample_to_tensors(self.dataset[index])
