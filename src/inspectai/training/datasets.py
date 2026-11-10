"""Torch datasets for autoencoder training (indices into MVTec train/good)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.training.preprocess import image_to_float01


class FloatImageSubset(Dataset[torch.Tensor]):
    """Resize uint8 MVTec images to float32 [0, 1] at a model input size."""

    def __init__(
        self,
        dataset: MVTecTorchDataset,
        indices: Sequence[int],
        input_size: int,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.dataset[self.indices[index]]
        return image_to_float01(item["image"], self.input_size)
