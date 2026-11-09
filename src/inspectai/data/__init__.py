"""Public data-pipeline API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.transforms import IdentityTransform
from inspectai.data.types import (
    MVTecError,
    MVTecLayoutError,
    MVTecPairingError,
    MVTecSample,
)

if TYPE_CHECKING:
    from inspectai.data.torch_dataset import MVTecTorchDataset

__all__ = [
    "IdentityTransform",
    "MVTecDataset",
    "MVTecError",
    "MVTecLayoutError",
    "MVTecPairingError",
    "MVTecSample",
    "MVTecTorchDataset",
]


def __getattr__(name: str) -> Any:
    if name == "MVTecTorchDataset":
        from inspectai.data.torch_dataset import MVTecTorchDataset

        return MVTecTorchDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
