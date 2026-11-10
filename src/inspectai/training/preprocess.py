"""Model-specific tensors. Does not change MVTecDataset native loading."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def image_to_float01(image_uint8_chw: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 CHW -> float32 [0, 1] CHW at ``size`` (bilinear)."""
    x = image_uint8_chw.float() / 255.0
    return F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0]


def mask_to_native_or_size(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """HW mask -> HW at ``size`` with nearest-neighbor resize."""
    if tuple(mask.shape[-2:]) == size:
        return mask.float()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(x, size=size, mode="nearest")
    return out[0, 0]


def upsample_map(score_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """HW float map -> native HW (bilinear)."""
    x = score_map.float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)[0, 0]
