# Quickstart

## Easiest path

```bash
./scripts/install.sh
./scripts/run_gui.sh
```

Windows PowerShell:

```powershell
./scripts/install.ps1
./scripts/run_gui.ps1
```

Then open:

`http://127.0.0.1:7777`

## Portable Bundle Build

macOS/Linux:

```bash
bash ./bundle/make_bundle.sh
```

Windows (PowerShell):

```powershell
./bundle/make_bundle.ps1
```

Outputs:

- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>.tar.gz` (macOS/Linux)
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>.zip` (Windows)

Note: Linux bundles can be significantly larger due to XGBoost/NCCL dependencies.
For Linux advanced users, `uv sync --extra pro` or a conda environment is recommended.

## CLI examples

```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```
