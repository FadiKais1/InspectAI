"""Inspect one image with a saved DINOv3 memory artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from inspectai.inference.inspector import Inspector
from inspectai.models.dinov3_anomaly import DINOWeightsError
from inspectai.visualization.overlays import save_rgb


def _anomaly_map_png(score_map: np.ndarray) -> np.ndarray:
    heat = np.asarray(score_map, dtype=np.float64)
    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        norm = np.zeros_like(heat)
    else:
        lo, hi = float(finite.min()), float(finite.max())
        norm = (heat - lo) / (hi - lo + 1e-8)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one image with Inspector")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/artifacts/dinov3_transistor"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/demo/example"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional explicit score threshold. Not estimated from official test labels.",
    )
    args = parser.parse_args()
    try:
        inspector = Inspector.load(args.artifact, device=args.device)
    except DINOWeightsError as exc:
        raise SystemExit(str(exc)) from exc

    with Image.open(args.image) as src:
        result = inspector.inspect(src.convert("RGB"), threshold=args.threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output / "anomaly_map.png", _anomaly_map_png(result.anomaly_map))
    save_rgb(args.output / "overlay.png", result.overlay)
    payload = result.summary()
    payload["source_image"] = str(args.image)
    payload["artifact"] = {
        "dir": str(args.artifact),
        "backbone": inspector.metadata.get("backbone"),
        "input_size": inspector.metadata.get("input_size"),
        "patch_size": inspector.metadata.get("patch_size"),
        "embedding_dim": inspector.metadata.get("embedding_dim"),
        "n_memory_patches": inspector.metadata.get("n_memory_patches"),
        "seed": inspector.metadata.get("seed"),
        "k": inspector.metadata.get("k"),
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output / 'anomaly_map.png'}")
    print(f"wrote {args.output / 'overlay.png'}")
    print(f"wrote {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
