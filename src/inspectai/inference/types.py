"""Structured outputs for reusable inspection inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectionResult:
    """Score and localization for one image. No default NORMAL/DEFECTIVE label."""

    anomaly_score: float
    anomaly_map: np.ndarray
    overlay: np.ndarray
    latency_ms: float
    model_name: str
    input_hw: tuple[int, int]
    patch_grid: tuple[int, int]
    threshold: float | None = None

    def summary(self) -> dict:
        payload: dict = {
            "anomaly_score": float(self.anomaly_score),
            "latency_ms": float(self.latency_ms),
            "model_name": self.model_name,
            "input_hw": [int(self.input_hw[0]), int(self.input_hw[1])],
            "patch_grid": [int(self.patch_grid[0]), int(self.patch_grid[1])],
        }
        if self.threshold is not None:
            payload["threshold"] = float(self.threshold)
            payload["above_threshold"] = bool(self.anomaly_score >= self.threshold)
        return payload
