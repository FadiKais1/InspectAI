"""Export the locked M7 DINOv3 memory from train/good only. Not a new experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectai.inference.build import build_dinov3_memory_artifact
from inspectai.models.dinov3_anomaly import DINOWeightsError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit locked M7 DINOv3 memory on train/good and save an artifact"
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/artifacts/dinov3_transistor"),
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    try:
        meta = build_dinov3_memory_artifact(
            args.root,
            args.category,
            args.output,
            device=args.device,
        )
    except DINOWeightsError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
