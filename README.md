# ASCENDS

**A practical ML toolkit for scientists who want answers, not boilerplate.**

ASCENDS (Advanced data SCiEnce toolkit for Non-Data Scientists) helps research users run local machine learning on their CSV data through a GUI and CLI.

## What It Does

- Correlation analysis: Pearson, Spearman, MI, dCor
- Model training and prediction workflows
- Regression outputs with parity plots
- Classification backend support with confusion matrix generation
- Reproducible run artifacts in `runs/`

## Why It Exists

Many domain experts have high-value data but limited time for coding. ASCENDS provides a guided workflow for:

- selecting inputs and target columns
- training quick baseline models
- understanding model quality through clear metrics and plots

All processing runs locally.

## Quick Start

```bash
./install.sh
./run_gui.sh
```

Open: `http://127.0.0.1:7777`

Manual path (advanced users):

```bash
uv sync
uv run ascends gui
```

Pro profile (includes `xgboost` + `shap`):

```bash
uv sync --extra pro
uv run ascends gui
```

## Smoke Test

```bash
./test/test.sh
```

## Hyperparameter Tuning (Roadmap)

- Planned next step:
  - `standard`: quick/limited tuning presets
  - `pro`: expanded tuning + Optuna advanced mode

## Portable Bundle (No Setup on Target Machine)

Build on your machine:

```bash
bash ./bundle/make_bundle.sh standard
# or
bash ./bundle/make_bundle.sh pro
```

Windows (PowerShell):

```powershell
./bundle/make_bundle.ps1 -Profile standard
# or
./bundle/make_bundle.ps1 -Profile pro
```

- `standard`: lighter bundle (no `xgboost` / `shap`)
- `pro`: includes `xgboost` + `shap`

Release policy (current):

- `v0.3.0`: official/public release target is `pro`
- `standard`: experimental/internal until profile split UX is fully stabilized

This creates:
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<profile>.tar.gz` on macOS/Linux
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<profile>.zip` on Windows

Note: Linux `pro` bundles can be significantly larger because optional XGBoost dependencies include NVIDIA NCCL runtime packages.
On the target machine (same OS/arch family), unpack and run:

```bash
./launch_gui.sh
```

## Documentation

- Developer and internal details: [`README.dev.md`](/Users/ds6/ASCENDS/README.dev.md)

## License

MIT
