# InspectAI v1.0 release checklist

Do not treat this as a production-deployment certificate. InspectAI v1.0 is
a portfolio / research prototype evaluated on MVTec AD transistor.

## Engineering

- [x] `python -m pytest` passes (record count below)
- [x] `python -m ruff check .` is clean
- [x] CLI `--help` works for experiment and inference scripts
- [x] Package version is **1.0.0** (`pyproject.toml` and `inspectai.__version__`)
- [x] Working tree reviewed with `git status` before commit

Last recorded local run: **69 passed**; Ruff **All checks passed**.

## Documentation

- [x] README complete (problem, results, protocol, install, demo)
- [x] `docs/README.md` index
- [x] M4, M5, M6, M7 experiment notes
- [x] `docs/final_results.md` with locked metrics
- [x] Honest limitations; DINOv3 not claimed best at image AUROC
- [x] Qualitative photos **not** committed (MVTec license)

## Repository hygiene

- [x] `data/raw` gitignored (dataset not in git)
- [x] `outputs/experiments`, `outputs/artifacts`, `outputs/demo` gitignored
- [x] `.venv`, `__pycache__`, `.ipynb_checkpoints`, `.env` ignored
- [x] No Hugging Face tokens or secrets in the tree
- [x] Notebook cell-id noise restored rather than committed

## Runtime (manual)

- [x] DINOv3 memory artifact can be rebuilt from train/good
- [x] `Inspector.load` + `inspect` works with the saved artifact
- [x] `streamlit run app/streamlit_app.py` loads and inspects an image
- [x] Demo does not show NORMAL/DEFECTIVE by default

## Git (manual; do not automate the tag)

See the commands in the v1.0 wrap-up: commit on `main`, then annotated tag
`v1.0.0` only after you inspect `git status` and `git diff`.
