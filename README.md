# InspectAI

Industrial visual anomaly detection and quality inspection.

InspectAI is a research-oriented Python project for detecting visual defects
in industrial images. The long-term direction is a real-time inspection
system built around PyTorch computer-vision models.

## Current status

The repository is a src-layout scaffold plus transistor EDA
(`notebooks/01_mvtec_eda.ipynb`). There are no trained models, dataset
loaders, APIs, or dashboards yet. Reusable library code belongs under
`src/inspectai`; notebooks are for EDA, visualization, and analysis.

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

Requires Python 3.11 or newer. Runtime packages are pandas, NumPy, Pillow, and
Matplotlib. PyTorch is an optional extra (``pip install inspectai[torch]``) used
by the thin ``MVTecTorchDataset`` adapter; it is not required for NumPy loading.

## Streamlit demo

The interactive demo uses a **saved** DINOv3 memory artifact. It does not
rebuild the train/good memory bank.

1. Build the artifact (once):

```text
python scripts/build_dinov3_memory.py --root data/raw --category transistor --output outputs/artifacts/dinov3_transistor
```

2. Launch the app:

```text
streamlit run app/streamlit_app.py
```

3. Upload a PNG or JPEG image and click **Inspect Image**.
