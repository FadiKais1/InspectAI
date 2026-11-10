"""Compact convolutional autoencoder (Milestone 5 reconstruction baseline)."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Four-level encoder/decoder without skip connections or pretrained weights."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _conv_block(in_channels, 32)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = _conv_block(32, 64)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = _conv_block(64, 128)
        self.down3 = nn.MaxPool2d(2)
        self.enc4 = _conv_block(128, 256)
        self.down4 = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = _conv_block(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _conv_block(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = _conv_block(32, 32)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec4 = _conv_block(32, 32)
        self.head = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down1(self.enc1(x))
        z = self.down2(self.enc2(z))
        z = self.down3(self.enc3(z))
        z = self.down4(self.enc4(z))
        y = self.dec1(self.up1(z))
        y = self.dec2(self.up2(y))
        y = self.dec3(self.up3(y))
        y = self.dec4(self.up4(y))
        return torch.sigmoid(self.head(y))


def reconstruction_error_map(image: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean squared error across RGB -> 2D map. Accepts CHW or NCHW."""
    if image.ndim != recon.ndim:
        raise ValueError("image and recon must have the same rank")
    if image.ndim == 3:
        return torch.mean((image - recon) ** 2, dim=0)
    if image.ndim == 4:
        return torch.mean((image - recon) ** 2, dim=1)
    raise ValueError(f"Expected CHW or NCHW, got shape {tuple(image.shape)}")


def image_anomaly_score(error_map: torch.Tensor) -> torch.Tensor:
    """Spatial maximum. 2D map -> scalar; NHW -> (N,)."""
    if error_map.ndim == 2:
        return error_map.amax()
    if error_map.ndim == 3:
        return error_map.flatten(1).amax(dim=1)
    raise ValueError(f"Expected HW or NHW, got shape {tuple(error_map.shape)}")
