# ASCENDS

**A practical ML toolkit for scientists who want answers, not boilerplate.**

ASCENDS (Advanced data SCiEnce toolkit for Non-Data Scientists) helps research users run local machine learning on their CSV data through a GUI and CLI.

## What It Does

- Correlation analysis: Pearson, Spearman, MI, dCor
- Model training and prediction workflows
- Regression outputs with parity plots and SHAP feature importance
- Classification backend support with confusion matrix generation
- Reproducible run artifacts in `runs/`

All processing runs locally — no data leaves your machine.

---

## For End Users — Portable Bundle (No Setup Required)

Download the zip for your platform from the Releases page, unzip anywhere, and run:

| Platform | Launcher |
|----------|----------|
| Windows  | Double-click `launch_gui.bat` |
| macOS / Linux | `./launch_gui.sh` |

Then open `http://127.0.0.1:7777` in your browser.

> **First launch on Windows** may take 1–2 minutes while math libraries compile. Subsequent launches are fast.

No Python, no uv, no installation needed. Everything is bundled.

---

## For Developers — Quick Start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ascends gui
```

Or use the helper scripts:

```bash
# macOS / Linux
./scripts/install.sh
./scripts/run_gui.sh

# Windows (cmd)
scripts\install.bat
scripts\run_gui.bat

# Windows (PowerShell)
./scripts/install.ps1
./scripts/run_gui.ps1
```

Open: `http://127.0.0.1:7777`

---

## CLI Examples

```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```

---

## Building a Portable Bundle

```bash
# macOS / Linux
bash ./bundle/make_bundle.sh

# Windows (cmd)
bundle\make_bundle.bat

# Windows (PowerShell)
./bundle/make_bundle.ps1
```

Output:
- `dist/ASCENDS-v<version>-<YYYYMMDD>-linux.tar.gz`
- `dist/ASCENDS-v<version>-<YYYYMMDD>-macOS.tar.gz`
- `dist/ASCENDS-v<version>-<YYYYMMDD>-windows.zip`

The Windows bundle includes a full Python distribution — no dependencies needed on the target machine.
See [`docs/windows_bundle_handoff.md`](docs/windows_bundle_handoff.md) for architecture details.

> **Note:** Linux bundles can be significantly larger because XGBoost may pull NVIDIA NCCL runtime wheels. For Linux, `uv sync` in a dedicated environment is often preferred over the bundle.

---

## Smoke Test

```bash
# macOS / Linux
./test/test.sh

# Windows (PowerShell)
./test/test.ps1
```

---

## License

MIT
