# InspectAI

Industrial visual anomaly detection and quality inspection.

InspectAI is a research-oriented Python project for detecting visual defects
in industrial images. The long-term direction is a real-time inspection
system built around PyTorch computer-vision models.

## Current status

This repository is a **scaffold only**. There are no trained models, dataset
loaders, APIs, or dashboards yet. Reusable library code lives under
`src/inspectai`; notebooks are reserved for EDA, visualization, and analysis.

## Planned direction

- Dataset handling and evaluation for industrial inspection
- PyTorch models for visual anomaly detection and quality control
- Reproducible experiments via configs
- Inference suitable for later real-time use

## Repository structure

```text
src/inspectai/          Installable package (src-layout)
  data/                 Dataset contracts and I/O (later)
  models/               PyTorch model code (later)
  evaluation/           Metrics and inspection criteria (later)
  visualization/        Shared plotting helpers (later)
  inference/            Run-time prediction (later)
  utils/                Shared helpers
configs/                Data, model, and experiment configs
notebooks/              EDA and analysis only
scripts/                Thin CLI entrypoints
tests/                  Package tests
data/raw/               Original local datasets (not committed)
data/processed/         Derived local artifacts (not committed)
outputs/                Figures, checkpoints, experiment runs
```

Requires Python 3.11 or newer. Runtime ML dependencies are not declared yet.
