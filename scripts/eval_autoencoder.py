"""CLI: official test evaluation for the trained autoencoder. Does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectai.models.run_autoencoder import evaluate_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI M5 autoencoder test evaluation")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/experiments/m5_transistor_autoencoder/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/m5_transistor_autoencoder"),
    )
    args = parser.parse_args()
    metrics = evaluate_autoencoder(
        args.root,
        args.category,
        args.checkpoint,
        args.output,
        input_size=256,
    )
    out_json = args.output / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {args.output / 'scores.csv'}")
    print(f"wrote {args.output / 'analysis'}")


if __name__ == "__main__":
    main()
