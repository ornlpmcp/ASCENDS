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
│   ├── gui_correlation_routes.py
│   ├── gui_predict_routes.py
│   ├── gui_saved_run_routes.py
│   ├── gui_shap_routes.py
│   ├── gui_train_run_routes.py
│   ├── gui_train_select_routes.py
│   ├── gui_plotting.py
│   ├── gui_run_registry.py
│   └── utils/
├── ascends_server.py          # FastAPI app setup, shared wiring, page routes
├── templates/                 # Jinja templates
├── static/                    # CSS/JS/images + generated plots
├── examples/                  # Sample datasets
└── test/                      # Smoke scripts
```

The GUI backend is intentionally split by workflow. `ascends_server.py` owns app initialization, workspace manifests, shared `LAST_TRAIN` state, and router registration; workflow-specific request handling lives in the `ascends/gui_*_routes.py` modules.

## Runtime Flow (GUI)

1. Upload CSV in `Correlation` tab.
2. Select inputs/target and persist via workspace manifest.
3. Run `Train`:
   - Regression: metrics + parity plot
   - Classification: metrics + confusion matrix
4. Optionally run SHAP/feature importance.
5. Save model run into `runs/<name>/` and generate `report.html`.
6. Use `Predict` tab for new CSV scoring.

## Artifacts

Typical run artifacts:

- `model.joblib`
- `manifest.json`
- `metrics.csv`
- `parity_train.csv`, `parity_test.csv`, `parity_all.csv` (regression path)
- `predictions.csv`
- Plot images (`parity.png` or `confusion.png` depending on task/path)
- `report.html` for saved GUI runs

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

Create an OS-specific portable bundle. Windows and macOS bundles include a copied Python runtime plus the prebuilt package environment, so target users do not need Python or `uv`.

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

- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<arch>/`
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<arch>.zip`

Note: `uv` is required on the build machine but is not copied into the release bundle. Linux self-contained bundles are intentionally not supported right now; Linux power users should use `uv sync` or a dedicated conda environment instead.

Bundle usage on target machine:

```bash
# macOS
open ./launch_gui.command

# Linux
uv run ascends gui
```

Windows launchers are also generated:

- `launch_gui.bat` / `launch_cli.bat`
- `launch_gui.ps1`

## Known Status

- `parity-plot` crash path on macOS backend was fixed by forcing headless plotting in CLI.
- GUI workflow routes are split into focused router modules:
  - correlation
  - train selection
  - train execution
  - SHAP
  - saved run/report/delete
  - prediction
- Classification is enabled in GUI backend training path with:
  - `Accuracy`, `Precision`, `Recall`, `F1`
  - optional `ROC_AUC` for binary classification
  - confusion matrix image output
- SHAP/feature-importance flow is implemented in CLI and GUI
  (tree SHAP with permutation fallback).

## Product Roadmap (Active)

1. Move remaining shared domain logic out of GUI routers where it should be reused by CLI/core.
2. Improve classification consistency across CLI and GUI surfaces.
3. Add clearer UI rendering for classification metrics in Train view.
4. Hyperparameter tuning rollout:
   - expanded search + Optuna (advanced)
5. Tighten portable bundle behavior and documentation across platforms.

## Dataset References

- `examples/BostonHousing.csv` (regression)
- `examples/BostonHousing_test.csv` (regression inference)
- `examples/iris.csv` (classification)
- `examples/fatigue.csv` (materials-related sample)
