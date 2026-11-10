"""CLI: Milestone 6 PatchCore-style baseline. Do not retune after test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectai.models.patchcore import PatchCoreConfig
from inspectai.models.run_patchcore import run_patchcore


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI M6 PatchCore-style anomaly detection")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/m6_transistor_patchcore"),
    )
    args = parser.parse_args()
    metrics = run_patchcore(args.root, args.category, args.output, config=PatchCoreConfig())
    out_json = args.output / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {args.output / 'scores.csv'}")
    print(f"wrote {args.output / 'memory.json'}")
    print(f"wrote {args.output / 'analysis'}")


if __name__ == "__main__":
    main()
