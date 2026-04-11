# ASCENDS Roadmap

> Last updated: 2026-04-11 (v0.4.0)

## Design Philosophy

> **"Upload CSV → Train model → Understand results → Share."**
> Every feature must directly serve this flow. ASCENDS is built for domain experts, not data scientists — simplicity is a feature, not a limitation.

---

## v0.4.x — Stability & UX

- [ ] **Run Report** — Auto-generate `runs/<name>/report.html` on Save Model; add Report button in ML Models panel
  - Includes: metrics, parity/confusion plots, SHAP importance, rule-based interpretation text
  - Interpretation module: `ascends/core/interpret.py` — scenario-based rules for R², MAE, overfitting, F1, class imbalance, etc.
- [ ] **Loading spinner** — Show progress indicator during Train / SHAP / Correlation runs
- [ ] **Data quality diagnostics** — Auto-detect missing values, outliers, class imbalance; surface warnings before training
- [ ] **Feature alignment warning** — Warn user when prediction data has different features than training (`data.py:align_to_features`)
- [ ] **SHAP multi-class fix** — Average absolute SHAP values across classes for multi-class classification (`explain.py:save_default_shap_plot`)
- [ ] **Broad exception handling cleanup** — Replace `except Exception: pass` with `logger.warning()` throughout

---

## v0.5.0 — Architecture Cleanup

> Scheduled after Run Report to avoid the server file growing further before the split.

- [ ] **Remove core duplication** — Delete `_compute_correlations()` from `ascends_server.py`; import from `ascends/core/correlation.py` instead
- [ ] **Unify `task` representation** — Consistently apply `canonicalize_task()` at all entry points (server + CLI)
- [ ] **Split `ascends_server.py`** — Move correlation / train / predict into separate router files

---

## v0.6.0 — Model Insight

- [ ] **Baseline comparison** — Show performance vs. dummy model (mean prediction) to give context
- [ ] **Model comparison view** — Side-by-side view of multiple runs in `runs/`
- [ ] **Hyperparameter tuning** — Currently a placeholder; Optuna integration (keep UI simple — one "Tune" button)
- [ ] **Windows packaging improvements** — Stabilize bat scripts and bundle workflow (`docs/windows_handoff.md`)

---

## v0.7.0 — Workflow Expansion

> Features here must clear the design philosophy bar before implementation.

- [ ] **Time-series support** — Time-based split already exists in core; expose in GUI
- [ ] **Classification CLI `--with-proba`** — Output probability columns when model supports `predict_proba`
- [ ] **LLM interpretation (optional)** — Claude API integration for richer interpretation; offline rule-based fallback must remain the default
- [ ] **Frontend modernization** — Incremental TypeScript + Tailwind adoption (no Next.js)

---

## Completed

- [x] Remove duplicate matplotlib import `v0.4.0 · 2026-04-11`
- [x] Add 50 MB upload size limit `v0.4.0 · 2026-04-11`
- [x] Fix path traversal vulnerability (`/predict/download`) `v0.4.0 · 2026-04-11`
- [x] Remove duplicate top_k parsing block (was overwriting corr settings) `v0.4.0 · 2026-04-11`
- [x] Remove unreachable code after FileResponse return `v0.4.0 · 2026-04-11`
- [x] Remove duplicate `_unique_preserve()` definition `v0.4.0 · 2026-04-11`
- [x] Fix TOCTOU race condition in `_unique_run_name` `v0.4.0 · 2026-04-11`
- [x] Cap `LAST_TRAIN` cache at 20 entries `v0.4.0 · 2026-04-11`
- [x] SHAP dual-view (ASCENDS bar / default beeswarm) `v0.4.0 · 2026-04-11`
- [x] Classification GUI support `v0.4.0 · 2026-04-11`
