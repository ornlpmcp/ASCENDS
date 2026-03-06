# ASCENDS Developer Guide

This document is the internal/developer companion to `README.md`.

## Project Goal

ASCENDS is built for scientists and engineers who want local, fast ML workflows without heavy coding overhead.

Key product direction:

- Local/offline-first workflows
- GUI-first experience with CLI parity
- Portable desktop packaging roadmap (single default package)

## Current Architecture

```text
ASCENDS/
├── ascends/
│   ├── cli.py                 # Typer CLI entrypoint
│   ├── core/
│   │   ├── correlation.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── models.py
│   │   └── ...
│   └── utils/
├── ascends_server.py          # FastAPI GUI backend
├── templates/                 # Jinja templates
├── static/                    # CSS/JS/images + generated plots
├── examples/                  # Sample datasets
└── test/                      # Smoke scripts
```

## Runtime Flow (GUI)

1. Upload CSV in `Correlation` tab.
2. Select inputs/target and persist via workspace manifest.
3. Run `Train`:
   - Regression: metrics + parity plot
   - Classification: metrics + confusion matrix
4. Save model run into `runs/<name>/`.
5. Use `Predict` tab for new CSV scoring.

## Artifacts

Typical run artifacts:

- `model.joblib`
- `manifest.json`
- `metrics.csv`
- `parity_train.csv`, `parity_test.csv`, `parity_all.csv` (regression path)
- `predictions.csv`
- Plot images (`parity.png` or `confusion.png` depending on task/path)

## Development Setup

```bash
uv sync
```

## Common Commands

### Launch GUI

```bash
uv run ascends gui
```

### Correlation example

```bash
uv run ascends correlation \
  --csv examples/BostonHousing.csv \
  --target medv \
  --task r \
  --metrics pearson,spearman,mi,dcor \
  --view wide
```

### Regression train/predict example

```bash
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```

## Test Strategy

### Primary smoke test (recommended)

```bash
./test/test.sh
```

This covers:

1. Correlation CLI path
2. Regression training CLI path
3. Parity plot generation
4. Prediction path
5. Classification GUI-backend smoke (`test/classification_gui_smoke.py`)

### Windows smoke test

```powershell
./test/test.ps1
```

## Portable Bundle Build

Create an OS-specific portable bundle (includes `.venv`):

```bash
bash ./bundle/make_bundle.sh
```

Windows (recommended):

```bat
bundle\make_bundle.bat
```

Windows PowerShell (optional):

```powershell
./bundle/make_bundle.ps1
```

Outputs:

- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>/`
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>.tar.gz` (macOS/Linux)
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>.zip` (Windows)

Note: Linux bundle size is expected to be larger because XGBoost can pull NVIDIA NCCL runtime wheels.
For Linux power users, `uv sync` or a dedicated conda environment is recommended.

Bundle usage on target machine:

```bash
./launch_gui.sh
```

Windows launchers are also generated:

- `launch_gui.ps1` / `launch_cli.ps1`
- `launch_gui.bat`

## Known Status

- `parity-plot` crash path on macOS backend was fixed by forcing headless plotting in CLI.
- GUI `train/select` route is implemented and wired to template actions.
- Classification is enabled in GUI backend training path with:
  - `Accuracy`, `Precision`, `Recall`, `F1`
  - optional `ROC_AUC` for binary classification
  - confusion matrix image output
- SHAP/feature-importance flow is implemented in CLI and GUI
  (tree SHAP with permutation fallback).

## Product Roadmap (Active)

1. Improve classification consistency across CLI and GUI surfaces.
2. Add clearer UI rendering for classification metrics in Train view.
3. Hyperparameter tuning rollout:
   - expanded search + Optuna (advanced)
4. Move from browser-first to desktop-app packaging path.

## Dataset References

- `examples/BostonHousing.csv` (regression)
- `examples/BostonHousing_test.csv` (regression inference)
- `examples/iris.csv` (classification)
- `examples/fatigue.csv` (materials-related sample)
