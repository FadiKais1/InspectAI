# Agent instructions for InspectAI

InspectAI is a professional machine-learning and computer-vision portfolio
project. It is also used for learning and interview preparation. Prefer
clear, defensible engineering over clever shortcuts.

## Stack and layout

- Target Python **3.11** (package requires `>=3.11`).
- **PyTorch** will be the main deep-learning framework when modeling starts.
  Do not add other ML frameworks unless the change is discussed first.
- Reusable Python belongs under `src/inspectai`.
- Notebooks are only for EDA, visualization, and analysis. Do not put
  training loops, dataset loaders, or production inference in notebooks.
- Thin wrappers belong in `scripts/`. Logic stays in the package.
- Configs live under `configs/data/`, `configs/model/`, and
  `configs/experiment/`.

## Research and experiments

- Do not fabricate metrics, plots, or experiment results.
- Before starting a new ML experiment, define the hypothesis, baseline,
  and evaluation metrics in writing.
- Major architectural changes (new model families, training stacks, APIs,
  dashboards, or layout changes) should be explained first and approved
  before implementation.

## What not to add without an explicit request

- YOLO, PatchCore, DINO, or other model implementations
- Dataset loaders, training code, APIs, or dashboards
- Unnecessary application frameworks
- GitHub Actions workflows or Cursor `.mdc` rules (use this file instead)

Keep the project simple, modular, and research-oriented.
