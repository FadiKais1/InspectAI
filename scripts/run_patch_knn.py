"""CLI: frozen patch-kNN on an MVTec category. Writes metrics JSON only from a real run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectai.models.patch_knn import PatchKNNConfig
from inspectai.models.run_patch_knn import run_patch_knn


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI M4 frozen patch-kNN baseline")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/m4_transistor_patch_knn"),
    )
    args = parser.parse_args()
    metrics = run_patch_knn(args.root, args.category, args.output, config=PatchKNNConfig())
    out_json = args.output / "metrics.json"
    if out_json.is_file():
        print(f"left existing {out_json} unchanged")
    else:
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {out_json}")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output / 'scores.csv'}")
    print(f"wrote {args.output / 'analysis'}")


if __name__ == "__main__":
    main()
