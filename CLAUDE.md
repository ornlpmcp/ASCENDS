# CLAUDE.md — ASCENDS Project Guidelines

## Language
- **Conversations**: Korean is fine
- **Everything else must be in English**: code, comments, commit messages, documentation, file contents

## Project Purpose
ASCENDS (Advanced data SCiEnce toolkit for Non-Data Scientists) is a local ML toolkit for domain experts who want to run ML workflows on their CSV data without coding. It provides a web GUI and CLI.

**Core user flow**: Upload CSV → Select features → Train model → Understand results → Share

## Design Philosophy
> Simplicity is a feature. Every addition must directly serve the core user flow. ASCENDS is for domain experts, not data scientists — do not add complexity that breaks that contract.

## Key Files
- `ascends_server.py` — FastAPI server (~1800 lines; split planned in v0.5.0)
- `ascends/core/` — Framework-agnostic ML logic (train, predict, correlation, explain, data)
- `ascends/cli.py` — CLI entry point (Typer)
- `templates/` — Jinja2 HTML templates
- `static/` — CSS, JS, generated plots
- `docs/ROADMAP.md` — Prioritized feature roadmap with version/date tags

## Architecture Notes
- Core modules are framework-agnostic (pure sklearn + pandas) — keep them that way
- Server and CLI both call the same core functions
- Workspace state stored in `workspace/{ws_id}/manifest.json`
- Run artifacts stored in `runs/<name>/` (model.joblib, metrics.csv, plots)
- `LAST_TRAIN` dict caches last trained model per workspace (capped at 20 entries)

## Known Tech Debt (see docs/ROADMAP.md for full list)
- `ascends_server.py` has duplicate `_compute_correlations()` vs `ascends/core/correlation.py`
- `task` representation inconsistent: server/CLI uses `"r"`/`"c"`, core uses `"regression"`/`"classification"`
- Many `except Exception: pass` blocks need `logger.warning()` replacements

## Development
```bash
uv sync
uv run ascends gui        # start GUI at http://127.0.0.1:7777
uv run ascends --help     # CLI
./test/test.sh            # smoke test
```
