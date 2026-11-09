"""Frozen ImageNet patch features + kNN memory bank (Milestone 4 baseline).

Hyperparameters are a priori. Do not select them by scanning test AUROC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PatchKNNConfig:
    """Locked a priori for the transistor baseline (see configs/experiment/)."""

    backbone: str = "resnet18"
    feature_layer: str = "layer3"
    input_size: int = 512
    k_neighbors: int = 1
    max_memory_patches: int = 50_000
    memory_sample_seed: int = 42
    image_score: str = "spatial_max"


def imagenet_tensor(image_uint8_chw: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 CHW -> ImageNet-normalized float NCHW at ``size`` (bilinear)."""
    if image_uint8_chw.ndim == 3:
        x = image_uint8_chw.unsqueeze(0).float() / 255.0
    else:
        x = image_uint8_chw.float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = x.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = x.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (x - mean) / std


class FrozenResNetPatchEncoder(nn.Module):
    """Frozen torchvision ResNet, returning one intermediate spatial map."""

    def __init__(self, layer_name: str = "layer3") -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        if layer_name == "layer2":
            self.head = nn.Identity()
            self._use_layer3 = False
        elif layer_name == "layer3":
            self.head = backbone.layer3
            self._use_layer3 = True
        else:
            raise ValueError(f"Unsupported feature layer {layer_name!r}")
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        if self._use_layer3:
            feats = self.head(feats)
        return F.normalize(feats, p=2, dim=1)


class PatchKNNScorer:
    """Memory bank of train/good patch features; test scores are kNN distances."""

    def __init__(
        self,
        encoder: nn.Module,
        config: PatchKNNConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or PatchKNNConfig()
        if self.config.k_neighbors != 1:
            raise ValueError("This baseline locks k_neighbors=1 a priori")
        if self.config.image_score != "spatial_max":
            raise ValueError("Only image_score='spatial_max' is supported")
        self.device = device or torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.memory: np.ndarray | None = None

    @torch.no_grad()
    def extract_patches(self, image_uint8_chw: torch.Tensor) -> np.ndarray:
        x = imagenet_tensor(image_uint8_chw.to(self.device), self.config.input_size)
        feats = self.encoder(x)
        b, c, h, w = feats.shape
        if b != 1:
            raise ValueError("extract_patches expects a single image")
        patches = feats[0].permute(1, 2, 0).reshape(h * w, c).cpu().numpy()
        return np.asarray(patches, dtype=np.float32)

    def fit(self, images: list[torch.Tensor]) -> None:
        banks = [self.extract_patches(image) for image in images]
        memory = np.concatenate(banks, axis=0)
        rng = np.random.default_rng(self.config.memory_sample_seed)
        cap = self.config.max_memory_patches
        if memory.shape[0] > cap:
            idx = rng.choice(memory.shape[0], size=cap, replace=False)
            memory = memory[idx]
        self.memory = memory

    def _patch_distances(self, patches: np.ndarray) -> np.ndarray:
        if self.memory is None:
            raise RuntimeError("Call fit() before scoring")
        mem = self.memory.astype(np.float32, copy=False)
        dmin = np.full(patches.shape[0], np.inf, dtype=np.float32)
        q_step, m_step = 32, 1024
        for i in range(0, patches.shape[0], q_step):
            query = patches[i : i + q_step]
            local = np.full(query.shape[0], np.inf, dtype=np.float32)
            q2 = np.sum(query * query, axis=1, keepdims=True)
            for j in range(0, mem.shape[0], m_step):
                bank = mem[j : j + m_step]
                m2 = np.sum(bank * bank, axis=1, keepdims=True).T
                dist_sq = np.clip(q2 + m2 - 2.0 * (query @ bank.T), 0.0, None)
                local = np.minimum(local, np.sqrt(dist_sq).min(axis=1))
            dmin[i : i + q_step] = local
        return dmin.astype(np.float64)

    def score_image(
        self, image_uint8_chw: torch.Tensor, native_hw: tuple[int, int]
    ) -> tuple[float, np.ndarray]:
        patches = self.extract_patches(image_uint8_chw)
        dist = self._patch_distances(patches)
        side = int(np.sqrt(dist.size))
        if side * side != dist.size:
            raise RuntimeError("Feature map is not square; unexpected encoder output")
        patch_map = dist.reshape(side, side)
        native_h, native_w = native_hw
        grid = torch.from_numpy(patch_map)[None, None].float()
        up = F.interpolate(grid, size=(native_h, native_w), mode="bilinear", align_corners=False)
        score_map = up[0, 0].numpy()
        image_score = float(patch_map.max())
        return image_score, score_map
