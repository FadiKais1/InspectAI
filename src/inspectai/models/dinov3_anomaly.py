"""Frozen DINOv3 ViT-S/16 patch embeddings for one-class anomaly detection.

Official weights
----------------
Model id: ``facebook/dinov3-vits16-pretrain-lvd1689m`` (ViT-S/16).

Load with Hugging Face Transformers (no DINOv2 substitute)::

    AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

Access / license
----------------
The checkpoint is gated under the DINOv3 License. Automated download fails
until you:

1. Open https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Accept the DINOv3 license (wait if access is still pending)
3. ``pip install transformers`` (InspectAI extra ``dinov3``)
4. ``huggingface-cli login`` with a token that has access to the repo

Token layout (DINOv3 ViT): ``[CLS] + num_register_tokens + patch tokens``.
Patch tokens are unflattened to a spatial grid; the CLS token is never scored.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from inspectai.models.patchcore import deterministic_candidate_pool, nearest_neighbor_l2
from inspectai.training.autoencoder import resolve_device
from inspectai.training.preprocess import upsample_map

OFFICIAL_DINOV3_VITS16 = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_LICENSE_URL = "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOWeightsError(RuntimeError):
    """Official DINOv3 weights are missing, gated, or Transformers is not installed."""


def dinov3_access_instructions(model_id: str = OFFICIAL_DINOV3_VITS16) -> str:
    return (
        "Official DINOv3 weights were not loaded. Do not substitute DINOv2 "
        "or another backbone. Required user action:\n"
        f"1. Open {DINOV3_LICENSE_URL} (model id: {model_id})\n"
        "2. Accept the DINOv3 License on the model card and wait until Hugging "
        "Face grants access\n"
        "3. pip install 'inspectai[dinov3]'  (or: pip install transformers)\n"
        "4. huggingface-cli login  (token must be allowed to download this repo)\n"
        "5. Re-run: python scripts/run_dinov3.py"
    )


def l2_normalize_embeddings(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize patch embeddings along the last dimension."""
    if embeddings.ndim < 1:
        raise ValueError("embeddings must have at least one dimension")
    norms = embeddings.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return embeddings / norms


def patch_tokens_to_grid(
    last_hidden_state: torch.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int,
    num_register_tokens: int,
) -> torch.Tensor:
    """Convert ``[CLS] + registers + patches`` to NCHW patch grid.

    ``last_hidden_state`` has shape (B, 1 + R + H_p * W_p, D).
    Returns (B, D, H_p, W_p) without the classification or register tokens.
    """
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must be (B, T, D)")
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")
    grid_h = height // patch_size
    grid_w = width // patch_size
    n_patches = grid_h * grid_w
    prefix = 1 + num_register_tokens
    expected = prefix + n_patches
    tokens = last_hidden_state.shape[1]
    if tokens != expected:
        raise ValueError(
            f"expected {expected} tokens (1 CLS + {num_register_tokens} registers "
            f"+ {n_patches} patches), got {tokens}"
        )
    patches = last_hidden_state[:, prefix:, :]
    batch, _, dim = patches.shape
    return patches.reshape(batch, grid_h, grid_w, dim).permute(0, 3, 1, 2).contiguous()


def load_official_dinov3(
    model_id: str = OFFICIAL_DINOV3_VITS16,
) -> tuple[object, nn.Module]:
    """Load official processor + frozen ViT. Never falls back to another model."""
    if model_id != OFFICIAL_DINOV3_VITS16:
        raise DINOWeightsError(
            f"Milestone 7 locks model_id={OFFICIAL_DINOV3_VITS16!r}; got {model_id!r}"
        )
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise DINOWeightsError(dinov3_access_instructions(model_id)) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    except Exception as exc:
        raise DINOWeightsError(dinov3_access_instructions(model_id) + f"\n\nCause: {exc}") from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


@dataclass
class DINOv3AnomalyConfig:
    input_size: int = 512
    candidate_pool_size: int = 50_000
    max_memory_patches: int = 10_000
    seed: int = 42
    k: int = 1
    model_id: str = OFFICIAL_DINOV3_VITS16
    patch_size: int = 16
    num_register_tokens: int = 4


class FrozenDINOv3PatchEncoder(nn.Module):
    """Frozen DINOv3 ViT that returns L2-normalized NCHW patch embeddings."""

    def __init__(
        self,
        processor: object,
        model: nn.Module,
        *,
        input_size: int = 512,
        patch_size: int = 16,
        num_register_tokens: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.model = model
        self.input_size = input_size
        self.patch_size = patch_size
        self.device = device or resolve_device()
        if num_register_tokens is None:
            num_register_tokens = int(getattr(model.config, "num_register_tokens", 4))
        self.num_register_tokens = num_register_tokens
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def preprocess(self, image_uint8_chw: torch.Tensor) -> torch.Tensor:
        """Official processor normalization at locked ``input_size`` (not 224)."""
        if image_uint8_chw.ndim != 3:
            raise ValueError("image must be CHW uint8")
        from PIL import Image

        pil_hwc = image_uint8_chw.permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(pil_hwc).convert("RGB")
        pil = pil.resize((self.input_size, self.input_size), Image.BICUBIC)
        tensors = self.processor(images=pil, return_tensors="pt", do_resize=False)
        pixel_values = tensors["pixel_values"]
        return pixel_values.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        hidden = outputs.last_hidden_state
        grid = patch_tokens_to_grid(
            hidden,
            height=self.input_size,
            width=self.input_size,
            patch_size=self.patch_size,
            num_register_tokens=self.num_register_tokens,
        )
        return l2_normalize_embeddings(grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DINOv3PatchScorer:
    """k=1 L2 to a deterministically subsampled normal patch memory."""

    def __init__(
        self,
        encoder: FrozenDINOv3PatchEncoder,
        config: DINOv3AnomalyConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or DINOv3AnomalyConfig()
        if self.config.k != 1:
            raise ValueError("this experiment uses k=1")
        if self.config.model_id != OFFICIAL_DINOV3_VITS16:
            raise ValueError(f"this experiment locks {OFFICIAL_DINOV3_VITS16}")
        self.memory: np.ndarray | None = None
        self.n_candidate_patches: int | None = None
        self.n_pool_patches: int | None = None
        self.grid_hw: tuple[int, int] | None = None

    def extract_feature_map(self, image_uint8_chw: torch.Tensor) -> torch.Tensor:
        pixel_values = self.encoder.preprocess(image_uint8_chw)
        with torch.no_grad():
            feats = self.encoder(pixel_values)
        if feats.ndim != 4 or feats.shape[0] != 1:
            raise ValueError("encoder must return a single-image NCHW map")
        return feats

    def extract_patches(self, image_uint8_chw: torch.Tensor) -> np.ndarray:
        feats = self.extract_feature_map(image_uint8_chw)
        _b, channels, height, width = feats.shape
        self.grid_hw = (height, width)
        patches = feats[0].permute(1, 2, 0).reshape(height * width, channels).cpu().numpy()
        return np.asarray(patches, dtype=np.float32)

    def fit(
        self,
        images: list[torch.Tensor],
        *,
        log: Callable[[str], None] | None = print,
    ) -> None:
        if log is not None:
            log("extracting training features")
        banks = []
        n_images = len(images)
        for i, image in enumerate(images, start=1):
            banks.append(self.extract_patches(image))
            if log is not None and (i % 10 == 0 or i == n_images):
                log(f"extracting training features {i}/{n_images}")
        candidates = np.concatenate(banks, axis=0)
        self.n_candidate_patches = int(candidates.shape[0])
        if log is not None:
            log(f"number of raw candidate patches: {self.n_candidate_patches}")

        pool_idx = deterministic_candidate_pool(
            self.n_candidate_patches,
            self.config.candidate_pool_size,
            seed=self.config.seed,
        )
        pool = candidates[pool_idx]
        self.n_pool_patches = int(pool.shape[0])
        if log is not None:
            log(f"candidate pre-pool completed: {self.n_pool_patches}")

        cap = min(self.config.max_memory_patches, pool.shape[0])
        mem_idx = deterministic_candidate_pool(pool.shape[0], cap, seed=self.config.seed)
        self.memory = pool[mem_idx]
        if log is not None:
            log(f"final memory size: {self.memory.shape[0]}")

    def load_fitted_memory(
        self,
        memory: np.ndarray,
        *,
        grid_hw: tuple[int, int],
        n_candidate_patches: int | None = None,
        n_pool_patches: int | None = None,
    ) -> None:
        """Attach a previously saved memory bank. Does not refit from images."""
        bank = np.asarray(memory, dtype=np.float32)
        if bank.ndim != 2:
            raise ValueError("memory must be (N, D)")
        self.memory = bank
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.n_candidate_patches = n_candidate_patches
        self.n_pool_patches = n_pool_patches

    def score_image(
        self,
        image_uint8_chw: torch.Tensor,
        native_hw: tuple[int, int] | None = None,
    ) -> tuple[float, np.ndarray]:
        if self.memory is None:
            raise RuntimeError("call fit() before score_image()")
        patches = self.extract_patches(image_uint8_chw)
        distances = nearest_neighbor_l2(patches, self.memory)
        height, width = self.grid_hw  # type: ignore[misc]
        patch_map = distances.reshape(height, width)
        if native_hw is None:
            native_hw = (int(image_uint8_chw.shape[1]), int(image_uint8_chw.shape[2]))
        anomaly_map = upsample_map(
            torch.from_numpy(patch_map.astype(np.float32)),
            native_hw,
        ).numpy()
        image_score = float(np.max(patch_map))
        return image_score, anomaly_map

    def memory_report(self) -> dict:
        if self.memory is None:
            raise RuntimeError("call fit() first")
        return {
            "backbone": self.config.model_id,
            "input_size": self.config.input_size,
            "patch_size": self.config.patch_size,
            "grid_hw": list(self.grid_hw) if self.grid_hw else None,
            "embedding_dim": int(self.memory.shape[1]),
            "n_candidate_patches": self.n_candidate_patches,
            "n_pool_patches": self.n_pool_patches,
            "n_memory_patches": int(self.memory.shape[0]),
            "candidate_pool_size": self.config.candidate_pool_size,
            "max_memory_patches": self.config.max_memory_patches,
            "seed": self.config.seed,
            "k": self.config.k,
            "selection": "deterministic_random",
        }


def upsample_anomaly_map(patch_map: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    return upsample_map(torch.from_numpy(patch_map.astype(np.float32)), native_hw).numpy()
