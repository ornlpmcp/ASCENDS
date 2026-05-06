# Changelog

## 0.5.0 - 2026-05-06

### Added

- Added self-contained Windows/macOS bundle launchers that run from bundled Python without requiring target users to install Python or `uv`.
- Added macOS `launch_gui.command` for Finder double-click startup.
- Added browser auto-open support for the GUI via `ascends gui --open-browser`.

### Changed

- Bumped the project version to 0.5.0 after the GUI route modularization work.
- Made SHAP beeswarm the default GUI view when Tree SHAP is available.
- Disabled reload mode in end-user launcher scripts.
- Updated portable bundle filenames to include OS and architecture.
- Updated bundle documentation to make `uv` a build-time tool only for release bundles.

## 0.4.3 - 2026-05-05

### Fixed

- Fixed prediction feature alignment so prediction CSVs are aligned to the saved training feature order.
- Added regression coverage for prediction alignment and positive MAE reporting.

### Security

- Updated locked vulnerable dependencies reported by `pip-audit`.
- Kept dependency churn limited to the vulnerable packages and the required FastAPI/Starlette compatibility path.

### Changed

- Split the monolithic FastAPI server into focused GUI router modules:
  - correlation routes
  - prediction routes
  - SHAP routes
  - saved run, report, and delete routes
  - train selection route
  - train execution route
- Moved plotting helpers and saved-run registry helpers out of `ascends_server.py`.
- Reduced `ascends_server.py` to app setup, shared manifest wiring, page routes, and router registration.
- Added a Starlette 1.0-compatible template wrapper while preserving existing template call sites.
- Excluded `archive/lunch_and_learn` from version control and removed the local archive copy.

## 0.4.2 - 2026-04-11

### Added

- Added loading spinner overlay for slow GUI operations such as Train, SHAP, and Correlation.

### Fixed

- Reworked Windows portable bundle startup around a bundled `uv.exe`.
- Bundled a Python distribution for Windows to reduce target-machine setup friction.
- Patched Windows bundle issues related to numba/MAX_PATH and startup reliability.

### Changed

- Updated README bundle instructions and current developer workflow.
- Removed obsolete Windows handoff documentation after the bundle rewrite.

## 0.4.1 - 2026-04-11

### Added

- Added saved-run `report.html` generation with rule-based interpretation.
- Added live report preview without requiring Save Model first.
- Included metrics, parity/confusion plots, SHAP importance, and interpretation text in reports.

## 0.4.0 - 2026-04-11

### Added

- Added classification GUI backend support with confusion matrix output.
- Added SHAP dual-view support for ASCENDS bar plots and default SHAP plots.
- Added a 50 MB CSV upload size limit.

### Fixed

- Fixed `/predict/download` path traversal risk.
- Fixed duplicate correlation `top_k` parsing that could overwrite settings.
- Removed unreachable response code and duplicate helper definitions.
- Fixed saved-run naming race condition.
- Capped the in-memory `LAST_TRAIN` cache at 20 entries.

## 0.3.0

### Changed

- Stabilized the standard GUI startup profile.
- Updated bundle naming to include version and build date.
- Simplified dependency and script workflows around the default `uv sync` path.
- Moved install and run scripts into `scripts/`.

## Initial Scaffold

### Added

- Created the initial FastAPI GUI and CLI structure.
