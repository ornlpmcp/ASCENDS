# ASCENDS Roadmap

> Last updated: 2026-05-05 (v0.4.3)

## Design Philosophy

> **"Upload CSV → Train model → Understand results → Share."**
> Every feature must directly serve this flow. ASCENDS is built for domain experts, not data scientists — simplicity is a feature, not a limitation.

---

## v0.4.x — Stability & UX

- [x] **Run Report** — Auto-generate `runs/<name>/report.html` on Save Model; add Report button in ML Models panel `v0.4.1 · 2026-04-11`
  - Includes: metrics, parity/confusion plots, SHAP importance, rule-based interpretation text
  - Interpretation module: `ascends/core/interpret.py` — scenario-based rules for R², MAE, overfitting, F1, class imbalance, etc.
- [x] **Loading spinner** — Show progress indicator during Train / SHAP / Correlation runs `v0.4.2 · 2026-04-11`
- [x] **Prediction feature alignment fix** — Align prediction CSVs to saved training feature order and add regression coverage `v0.4.3 · 2026-05-05`
- [x] **Dependency security refresh** — Update vulnerable locked dependencies with minimal churn and preserve Starlette template compatibility `v0.4.3 · 2026-05-05`
- [ ] **Data quality diagnostics** — Auto-detect missing values, outliers, class imbalance; surface warnings before training
- [ ] **Feature alignment warning** — Warn user when prediction data has different features than training (`data.py:align_to_features`)
- [ ] **SHAP multi-class fix** — Average absolute SHAP values across classes for multi-class classification (`explain.py:save_default_shap_plot`)
- [ ] **Broad exception handling cleanup** — Replace `except Exception: pass` with `logger.warning()` throughout

---

## v0.5.0 — Architecture Cleanup

> Most GUI route splitting was completed in v0.4.3. Remaining work should focus on shared domain logic, typing, and small testable helpers.

- [x] **Split `ascends_server.py`** — Move correlation / train / predict flows into separate router files `v0.4.3 · 2026-05-05`
- [x] **Move plotting/run-registry helpers** — Move GUI plotting and saved-run registry helpers out of `ascends_server.py` `v0.4.3 · 2026-05-05`
- [ ] **Promote correlation domain logic** — Move correlation computation from GUI router into `ascends/core/correlation.py` and share with CLI
- [ ] **Unify `task` representation** — Consistently apply `canonicalize_task()` at all entry points (server + CLI)
- [ ] **App factory cleanup** — Convert `ascends_server.py` wiring into a cleaner app-factory pattern if needed for tests/packaging

---

## v0.6.0 — Model Insight

- [ ] **Baseline comparison** — Show performance vs. dummy model (mean prediction) to give context
- [ ] **Model comparison view** — Side-by-side view of multiple runs in `runs/`
- [ ] **Hyperparameter tuning** — Currently a placeholder; Optuna integration (keep UI simple — one "Tune" button)
- [ ] **Packaging follow-up** — Tighten macOS/Linux bundle expectations and keep Windows double-click bundle fully self-contained

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
