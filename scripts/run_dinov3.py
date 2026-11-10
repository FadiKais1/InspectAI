"""CLI: Milestone 7 frozen DINOv3 patch anomaly detection. Do not retune after test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectai.models.dinov3_anomaly import DINOv3AnomalyConfig, DINOWeightsError
from inspectai.models.run_dinov3 import run_dinov3


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI M7 frozen DINOv3 patch AD")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/m7_transistor_dinov3"),
    )
    args = parser.parse_args()
    try:
        metrics = run_dinov3(args.root, args.category, args.output, config=DINOv3AnomalyConfig())
    except DINOWeightsError as exc:
        raise SystemExit(str(exc)) from exc
    out_json = args.output / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {args.output / 'scores.csv'}")
    print(f"wrote {args.output / 'memory.json'}")
    print(f"wrote {args.output / 'analysis'}")


if __name__ == "__main__":
    main()
