# Quickstart

## Easiest path

```bash
./install.sh
./run_gui.sh
```

Then open:

`http://127.0.0.1:7777`

## Profiles

- `standard`: lighter install/bundle (no `xgboost` / `shap`)
- `pro`: includes `xgboost` + `shap`

Install `pro` profile:

```bash
./install.sh pro
./run_gui.sh
```

Release policy (current):

- `v0.3.0` official/public recommendation: `pro`
- `standard` is currently internal/experimental

## Portable Bundle Build

macOS/Linux:

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

Outputs:

- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<profile>.tar.gz` (macOS/Linux)
- `dist/ASCENDS-v<version>-<YYYYMMDD>-<OS>-<profile>.zip` (Windows)

Note: Linux `pro` bundles can be significantly larger due to XGBoost/NCCL dependencies.

## CLI examples

```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```
