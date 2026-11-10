"""CLI: train the Milestone 5 conv autoencoder on transistor train/good only."""

from __future__ import annotations

import argparse
from pathlib import Path

from inspectai.data.mvtec import MVTecDataset
from inspectai.data.torch_dataset import MVTecTorchDataset
from inspectai.training.autoencoder import AutoencoderTrainConfig, train_autoencoder
from inspectai.training.datasets import FloatImageSubset
from inspectai.training.split import train_val_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI M5 convolutional autoencoder")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/m5_transistor_autoencoder"),
    )
    args = parser.parse_args()

    train_np = MVTecDataset(args.root, args.category, "train")
    train_ds = MVTecTorchDataset(train_np)
    cfg = AutoencoderTrainConfig()
    train_idx, val_idx = train_val_indices(
        len(train_ds),
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )
    train_subset = FloatImageSubset(train_ds, train_idx, cfg.input_size)
    val_subset = FloatImageSubset(train_ds, val_idx, cfg.input_size)
    history = train_autoencoder(train_subset, val_subset, args.output, cfg)
    print(f"best_epoch={history['best_epoch']} best_val_loss={history['best_val_loss']:.6f}")
    print(f"wrote {args.output / 'best.pt'}")
    print(f"wrote {args.output / 'training_history.json'}")


if __name__ == "__main__":
    main()
