"""Lightweight Milestone 5 tests. Do not train the full transistor model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from inspectai.models.autoencoder import (
    ConvAutoencoder,
    image_anomaly_score,
    reconstruction_error_map,
)
from inspectai.training.autoencoder import AutoencoderTrainConfig, train_autoencoder
from inspectai.training.split import train_val_indices


def test_split_is_deterministic_and_covers_all() -> None:
    a_train, a_val = train_val_indices(213, train_fraction=0.8, seed=42)
    b_train, b_val = train_val_indices(213, train_fraction=0.8, seed=42)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_val, b_val)
    assert len(a_train) == 170
    assert len(a_val) == 43
    assert set(a_train.tolist()).isdisjoint(a_val.tolist())
    assert len(set(a_train.tolist()) | set(a_val.tolist())) == 213


def test_split_changes_with_seed() -> None:
    t1, _ = train_val_indices(20, train_fraction=0.8, seed=42)
    t2, _ = train_val_indices(20, train_fraction=0.8, seed=0)
    assert not np.array_equal(t1, t2)


def test_forward_shape_range_and_batch() -> None:
    model = ConvAutoencoder()
    model.eval()
    x = torch.rand(4, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_mse_loss_and_cpu_backward() -> None:
    model = ConvAutoencoder()
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    loss = nn.MSELoss()(y, x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert float(loss.detach()) >= 0.0


def test_anomaly_map_and_score() -> None:
    image = torch.zeros(3, 8, 8)
    recon = torch.zeros(3, 8, 8)
    recon[:, 0, 0] = 1.0
    err = reconstruction_error_map(image, recon)
    assert err.shape == (8, 8)
    score = image_anomaly_score(err)
    assert score.ndim == 0
    assert float(score) == pytest.approx(float(err.max()))


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    images = torch.rand(4, 3, 32, 32)

    class TinySubset(Dataset):
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, index: int) -> torch.Tensor:
            return images[index]

    cfg = AutoencoderTrainConfig(
        input_size=32,
        seed=42,
        train_fraction=0.75,
        batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        patience=5,
    )
    history = train_autoencoder(TinySubset(), TinySubset(), tmp_path, cfg, device=torch.device("cpu"))
    ckpt = tmp_path / "best.pt"
    assert ckpt.is_file()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ConvAutoencoder()
    model.load_state_dict(blob["model_state_dict"])
    with torch.no_grad():
        out = model(images[:1])
    assert out.shape == (1, 3, 32, 32)
    assert (tmp_path / "training_history.json").is_file()
    assert history["stopped_epoch"] == 1


def test_dataloader_batch_forward() -> None:
    model = ConvAutoencoder()
    loader = DataLoader(TensorDataset(torch.rand(5, 3, 32, 32)), batch_size=2)
    batch = next(iter(loader))[0]
    y = model(batch)
    assert y.shape == batch.shape
