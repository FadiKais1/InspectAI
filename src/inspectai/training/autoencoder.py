"""Train a convolutional autoencoder on normal images only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inspectai.models.autoencoder import ConvAutoencoder


@dataclass(frozen=True)
class AutoencoderTrainConfig:
    input_size: int = 256
    seed: int = 42
    train_fraction: float = 0.8
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_epochs: int = 30
    patience: int = 5


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        recon = model(batch)
        loss = criterion(recon, batch)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * batch.shape[0]
        n += int(batch.shape[0])
    return total / max(n, 1)


def train_autoencoder(
    train_subset: Dataset[torch.Tensor],
    val_subset: Dataset[torch.Tensor],
    output_dir: Path,
    config: AutoencoderTrainConfig,
    *,
    device: torch.device | None = None,
) -> dict:
    """Fit on train_subset, early-stop on val reconstruction MSE. No test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()
    set_seed(config.seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    ckpt_path = output_dir / "best.pt"

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch_loss(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = _epoch_loss(model, val_loader, criterion, device, None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": asdict(config),
                },
                ckpt_path,
            )
        else:
            wait += 1
            if wait >= config.patience:
                break

    history_path = output_dir / "training_history.json"
    payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "stopped_epoch": len(history["train_loss"]),
        "device": str(device),
        "config": asdict(config),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
