# ASCENDS (Advanced data SCiEnce toolkit for Non-Data Scientists)

ASCENDS is a lightweight, open-source toolkit for data-driven materials design.
It provides both a **command-line interface (CLI)** and a **web-based GUI** built
on **FastAPI**, enabling non-experts to perform correlation analysis,
machine-learning model training, and visualization without complex scripting.

---

## 🧭 Features (current alpha)

| Category | Description |
|-----------|--------------|
| **Core analytics** | Correlation metrics (PCC, MIC, SHAP) for quick feature–target relationships |
| **Model training** | Regression/classification using XGBoost, SciKeras, and scikit-learn |
| **Parity plots** | Automated training vs test visualization for regression tasks |
| **GUI** | Modern web app powered by FastAPI + HTML templates |
| **CLI** | Simple Typer-based commands for batch workflows |
| **Train history (Phase 3b)** | Lightweight persistent log of all training runs (dataset, target, model, metrics, timestamp) |
| **Auto-suggest model names** | Default save name = `{dataset}_{target}_{YYYYMMDD_HHMM}` |
| **Config-free launch** | Self-contained — no database or conda environment required |
| **Cross-platform** | Runs on macOS, Linux, and Windows (WSL compatible) |

---

## 🚀 Quick start

### 1. Environment setup (with `uv`)
```bash
uv sync
```

### 2. Launch the GUI
```bash
uv run ascends gui
```
The web interface runs locally (default `http://127.0.0.1:7777`).

### 3. Run a quick CLI correlation
```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
```

---

## 🧠 Using the GUI

1. Open your browser to the displayed URL.
2. Upload a CSV dataset.
3. Choose a **target column**.
4. Select a **model type** and click **Train**.
5. Review results in:
   - **Test History** → log of all runs (Phase 3b feature)
   - **Saved Models** → manually saved models (`runs/`)
6. Download parity plots and model files directly.

---

## 💾 File & directory layout

```text
ASCENDS/
├── ascends/               # Core logic (CLI + analytics)
│   ├── cli.py             # Command-line interface (Typer)
│   ├── core/              # Correlation, training, plotting modules
│   └── utils/             # Helper scripts
├── ascends_server.py      # FastAPI app entry point (GUI backend)
├── templates/             # HTML templates for GUI
│   └── train.html
├── static/                # Client-side assets
└── runs/                  # Saved models and results (per session)
```

Additional files created at runtime:
```text
~/.ascends/train_history.json   # Append-only log of all training sessions
~/.ascends/cache/               # Temporary intermediate data
```

---

## 🧹 Maintenance & disk space

ASCENDS may accumulate temporary files under `.ascends/cache` and `runs/`.
A future **auto-cleanup** feature is planned (wishlist) to reclaim space on
startup or via `ascends cleanup --keep-days N`.

---

## 🧩 Development roadmap

| Phase | Focus | Status |
|--------|--------|--------|
| 3a | GUI parity plots & metrics panel | ✅ Complete |
| **3b** | Train history + save-name auto-suggest | 🚧 In progress |
| 3c | Model reload / test replay | ⏳ Planned |
| 4 | Batch prediction interface | ⏳ Planned |
| 5 | Auto-cleanup + storage management | 📝 Wishlist |

---

## 🧪 Example: training via GUI

After uploading `BostonHousing.csv` and selecting `medv` as the target:

| Field | Example |
|--------|---------|
| Model | XGBoost |
| Suggested name | `BostonHousing_medv_20251009_1200` |
| R² (test) | 0.92 |
| MAE | 0.32 |
| Saved path | `runs/BostonHousing_medv_20251009_1200/` |

Each run is automatically recorded in `train_history.json` with timestamp and metrics.

---

## 🧰 CLI reference

| Command | Description |
|----------|-------------|
| `ascends correlation` | Compute correlation metrics |
| `ascends parity-plot` | Generate parity plot for regression results |
| `ascends history` | (Phase 3b) Show training history summary |
| `ascends gui` | Launch web interface |

Use `--help` after any command for detailed options.

---

## 🧑‍💻 Contributing

Contributions are welcome!  
Please fork the repository and submit pull requests following these guidelines:

- Use clear commit messages (`feat(train): add auto-suggest`).
- Keep PRs small and focused.
- Run `ruff check` before submitting.

---

## 📜 License
MIT License © 2025 Oak Ridge National Laboratory & contributors

---

## 📞 Acknowledgments

ASCENDS was originally developed to support data-driven alloy design and
machine-learning–assisted materials research.  
Current modernization led by **Dongwon Shin**, ORNL, with ongoing enhancements
for reproducibility, UI modernization, and integrated ICME workflows.
