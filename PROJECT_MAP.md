# ASCENDS Project Map

This file records top-level project entries and when to use them.

| path | role | status | when-to-use |
| --- | --- | --- | --- |
| `README.md` | User-facing overview and install/run instructions | active | Start here for end-user usage, bundle notes, CLI examples, and smoke tests. |
| `quickstart.md` | Short setup and bundle command reference | active | Use for concise local setup and build commands. |
| `CHANGELOG.md` | Versioned release history | active | Use to understand what changed by release/version. |
| `PROJECT_MAP.md` | Top-level project inventory | active | Use to understand root files/directories and where new work belongs. |
| `LICENSE` | Project license | active | Use for licensing terms. |
| `pyproject.toml` | Python package metadata and dependency declarations | active | Use when changing package metadata, scripts, or dependency constraints. |
| `uv.lock` | Locked Python dependency graph | active | Use for reproducible installs and dependency/security updates. |
| `ascends/` | Python package source, core logic, CLI, and GUI route modules | active | Use for application logic changes. |
| `ascends_server.py` | FastAPI app setup, shared workspace manifest helpers, page routes, and router registration | active | Use for app wiring, not workflow-specific route logic. |
| `templates/` | Jinja HTML templates | active | Use for GUI markup changes. |
| `static/` | Static assets and generated GUI plot output | active | Use for CSS/JS/assets; generated workspace plots live under `static/workspace/`. |
| `examples/` | Sample datasets | active | Use for demos, smoke tests, and manual testing. |
| `test/` | Smoke and regression tests | active | Use for verification scripts and regression coverage. |
| `scripts/` | Install/run helper scripts | active | Use for local developer and user helper scripts. |
| `bundle/` | Portable bundle build scripts | active | Use when changing packaged distribution behavior. |
| `docs/` | Developer notes, roadmap, TODO, and design documentation | active | Use for internal documentation and planning. |
| `runs/` | Saved model runs | active/generated | Use for user-created saved models and reports; avoid committing run artifacts. |
| `workspace/` | GUI workspace manifests/uploads/intermediate artifacts | active/generated | Use for runtime GUI state; avoid committing workspace artifacts. |
| `dist/` | Bundle build outputs | generated | Use for local release artifacts; avoid committing generated bundles. |
| `__pycache__/` | Python bytecode cache | generated | Ignore or remove during cleanup; never use as source. |
