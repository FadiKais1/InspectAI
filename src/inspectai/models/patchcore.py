"""PatchCore-style multi-scale frozen features + greedy coreset memory.

Does not change Milestone 4 PatchKNNScorer behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from inspectai.models.patch_knn import imagenet_tensor


@dataclass(frozen=True)
class PatchCoreConfig:
    """Locked a priori for Milestone 6 (see configs/experiment/)."""

    backbone: str = "resnet18"
    feature_layers: tuple[str, ...] = ("layer2", "layer3")
    input_size: int = 512
    k_neighbors: int = 1
    max_memory_patches: int = 10_000
    seed: int = 42
    image_score: str = "spatial_max"
    projection_dim: int = 128


def align_and_concat(high_res: torch.Tensor, low_res: torch.Tensor) -> torch.Tensor:
    """Upsample ``low_res`` to ``high_res`` spatial size and concatenate channels.

    Tensors are NCHW. Interpolation is bilinear, align_corners=False.
    """
    if high_res.ndim != 4 or low_res.ndim != 4:
        raise ValueError("Expected NCHW feature maps")
    aligned = F.interpolate(
        low_res,
        size=high_res.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    fused = torch.cat([high_res, aligned], dim=1)
    return F.normalize(fused, p=2, dim=1)


class FrozenResNetMultiScaleEncoder(nn.Module):
    """Frozen ResNet18 returning layer2||upsampled-layer3 patch embeddings."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_layers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        layer2 = self.layer2(h)
        layer3 = self.layer3(layer2)
        return layer2, layer3

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer2, layer3 = self.forward_layers(x)
        return align_and_concat(layer2, layer3)


def greedy_farthest_point_indices(
    features: np.ndarray,
    n_select: int,
    *,
    seed: int = 42,
    projection_dim: int = 128,
) -> np.ndarray:
    """Deterministic greedy FPS in a fixed Gaussian random-projection space."""
    if features.ndim != 2:
        raise ValueError("features must be (N, D)")
    n_samples, dim = features.shape
    if n_select < 1:
        raise ValueError("n_select must be >= 1")
    if n_select >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((dim, projection_dim)).astype(np.float32)
    proj /= np.sqrt(float(projection_dim))
    projected = features.astype(np.float32, copy=False) @ proj
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.clip(norms, 1e-8, None)

    start = int(rng.integers(0, n_samples))
    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = start
    min_dist = np.full(n_samples, np.inf, dtype=np.float64)
    last = projected[start]
    for i in range(1, n_select):
        diff = projected - last
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        np.minimum(min_dist, dist, out=min_dist)
        min_dist[selected[:i]] = -np.inf
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        last = projected[nxt]
    return selected


def nearest_neighbor_l2(queries: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Min L2 distance from each query row to the memory bank."""
    mem = memory.astype(np.float32, copy=False)
    dmin = np.full(queries.shape[0], np.inf, dtype=np.float32)
    q_step, m_step = 32, 1024
    for i in range(0, queries.shape[0], q_step):
        query = queries[i : i + q_step]
        local = np.full(query.shape[0], np.inf, dtype=np.float32)
        q2 = np.sum(query * query, axis=1, keepdims=True)
        for j in range(0, mem.shape[0], m_step):
            bank = mem[j : j + m_step]
            m2 = np.sum(bank * bank, axis=1, keepdims=True).T
            dist_sq = np.clip(q2 + m2 - 2.0 * (query @ bank.T), 0.0, None)
            local = np.minimum(local, np.sqrt(dist_sq).min(axis=1))
        dmin[i : i + q_step] = local
    return dmin.astype(np.float64)


class PatchCoreScorer:
    """Train/good coreset memory; test scores are 1-NN distances."""

    def __init__(
        self,
        encoder: nn.Module,
        config: PatchCoreConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or PatchCoreConfig()
        if self.config.k_neighbors != 1:
            raise ValueError("Milestone 6 locks k_neighbors=1 a priori")
        if self.config.image_score != "spatial_max":
            raise ValueError("Only image_score='spatial_max' is supported")
        if self.config.feature_layers != ("layer2", "layer3"):
            raise ValueError("Milestone 6 locks feature_layers=('layer2', 'layer3')")
        self.device = device or torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.memory: np.ndarray | None = None
        self.coreset_indices: np.ndarray | None = None
        self.n_candidate_patches: int = 0

    @torch.no_grad()
    def extract_feature_map(self, image_uint8_chw: torch.Tensor) -> torch.Tensor:
        x = imagenet_tensor(image_uint8_chw.to(self.device), self.config.input_size)
        feats = self.encoder(x)
        if feats.ndim != 4 or feats.shape[0] != 1:
            raise ValueError("encoder must return a single-image NCHW map")
        return feats

    def extract_patches(self, image_uint8_chw: torch.Tensor) -> np.ndarray:
        feats = self.extract_feature_map(image_uint8_chw)
        _b, channels, height, width = feats.shape
        patches = feats[0].permute(1, 2, 0).reshape(height * width, channels).cpu().numpy()
        return np.asarray(patches, dtype=np.float32)

    def fit(self, images: list[torch.Tensor]) -> None:
        banks = [self.extract_patches(image) for image in images]
        candidates = np.concatenate(banks, axis=0)
        self.n_candidate_patches = int(candidates.shape[0])
        cap = self.config.max_memory_patches
        indices = greedy_farthest_point_indices(
            candidates,
            min(cap, candidates.shape[0]),
            seed=self.config.seed,
            projection_dim=self.config.projection_dim,
        )
        self.coreset_indices = indices
        memory = candidates[indices]
        norms = np.linalg.norm(memory, axis=1, keepdims=True)
        self.memory = memory / np.clip(norms, 1e-8, None)

    def score_image(
        self, image_uint8_chw: torch.Tensor, native_hw: tuple[int, int]
    ) -> tuple[float, np.ndarray]:
        if self.memory is None:
            raise RuntimeError("Call fit() before scoring")
        feats = self.extract_feature_map(image_uint8_chw)
        _b, _c, height, width = feats.shape
        patches = feats[0].permute(1, 2, 0).reshape(height * width, feats.shape[1]).cpu().numpy()
        dist = nearest_neighbor_l2(patches.astype(np.float32), self.memory)
        patch_map = dist.reshape(height, width)
        grid = torch.from_numpy(patch_map)[None, None].float()
        up = F.interpolate(grid, size=native_hw, mode="bilinear", align_corners=False)
        image_score = float(patch_map.max())
        return image_score, up[0, 0].numpy()

    def memory_metadata(self) -> dict:
        return {
            "n_candidate_patches": self.n_candidate_patches,
            "n_memory_patches": 0 if self.memory is None else int(self.memory.shape[0]),
            "feature_dim": 0 if self.memory is None else int(self.memory.shape[1]),
            "max_memory_patches": self.config.max_memory_patches,
            "projection_dim": self.config.projection_dim,
            "seed": self.config.seed,
            "coreset": "greedy_farthest_point",
        }
