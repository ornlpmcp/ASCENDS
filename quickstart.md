# Quickstart

## Before you start (Prerequisites)

- Common: Python 3.10+ and `uv`
- macOS: Homebrew is recommended for easy install
  - Example: `brew install python uv`
- Windows (PowerShell): install Python + uv first
  - Example: `winget install Python.Python.3.12` and `winget install astral-sh.uv`
- Linux: install Python + uv from your distro or the official uv installer

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

## CLI examples

```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```
