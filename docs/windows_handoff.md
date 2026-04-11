<!-- This file is a handoff note for debugging and improving ASCENDS Windows install/run/bundle behavior on a real Windows machine. -->
# ASCENDS Windows Handoff

## Purpose

This note is for continuing Windows-specific debugging on a real Windows machine with Codex.
The main goal is to make ASCENDS reliable for:

- install
- GUI run
- bundle build
- bundled GUI launch

without requiring the user to understand PowerShell policy details.

## Current Repo State

Relevant recent change:

- Commit `92fe69a`: added BAT-first Windows workflow

Current intended Windows entrypoints:

- install: `scripts\install.bat`
- GUI run: `scripts\run_gui.bat`
- bundle build: `bundle\make_bundle.bat`

PowerShell scripts still exist, but they are now intended to be optional:

- `scripts/install.ps1`
- `scripts/run_gui.ps1`
- `bundle/make_bundle.ps1`

## What Already Happened

The user tested on Windows and hit PowerShell execution policy errors such as:

- `.ps1 cannot be loaded`
- `is not digitally signed`
- `PSSecurityException`

This happened for:

- `scripts\run_gui.ps1`
- `bundle\make_bundle.ps1`

To reduce that problem, BAT wrappers were added.

## Important Reality Check

The BAT path is better, but it is not fully risk-free yet.

Why:

- `scripts\run_gui.bat` is pure cmd and should be the safest path
- `scripts\install.bat` is also pure cmd and should be safe
- `bundle\make_bundle.bat` still launches PowerShell internally with `-ExecutionPolicy Bypass`

That means:

- install/run may work even when direct `.ps1` execution fails
- bundle build may still fail in stricter enterprise environments

## Files To Inspect First

- [scripts/install.bat](/Users/ds6/ASCENDS/scripts/install.bat)
- [scripts/run_gui.bat](/Users/ds6/ASCENDS/scripts/run_gui.bat)
- [bundle/make_bundle.bat](/Users/ds6/ASCENDS/bundle/make_bundle.bat)
- [bundle/make_bundle.ps1](/Users/ds6/ASCENDS/bundle/make_bundle.ps1)
- [quickstart.md](/Users/ds6/ASCENDS/quickstart.md)
- [README.md](/Users/ds6/ASCENDS/README.md)

## Recommended Test Order On Windows

Run these from `cmd.exe` first, not PowerShell.

### 1. Environment sanity

```bat
where python
where py
where uv
python --version
py -3.11 --version
uv --version
```

Capture:

- which interpreter is actually used
- whether `uv` is on PATH

### 2. Fresh install path

```bat
cd C:\path\to\ASCENDS
scripts\install.bat
```

Expected:

- `.venv\Scripts\ascends.exe` created
- `uv sync` completes

If it fails, capture:

- full command output
- whether the failure is PATH, Python, wheel build, or network related

### 3. GUI run path

```bat
cd C:\path\to\ASCENDS
scripts\run_gui.bat
```

Expected:

- starts server at `http://127.0.0.1:7777`
- browser can open manually

If it fails, capture:

- exact console output
- whether `.venv\Scripts\ascends.exe` exists

### 4. Bundle build path

```bat
cd C:\path\to\ASCENDS
bundle\make_bundle.bat
```

Expected:

- creates `dist\ASCENDS-v0.3.0-<date>-windows.zip`

Known risk:

- this may still fail due to PowerShell restrictions, because the BAT wrapper still invokes `make_bundle.ps1`

### 5. Bundled app run path

Unzip the generated Windows bundle, then test:

```bat
launch_gui.bat
```

Expected:

- bundled GUI starts without local Python/uv setup

## Most Likely Failure Categories

### A. PowerShell policy / enterprise policy

Symptoms:

- script cannot be loaded
- not digitally signed
- execution disabled

Most affected path:

- `bundle\make_bundle.bat` -> `bundle\make_bundle.ps1`

Best next step:

- replace Windows bundle build with a true cmd-only path if needed
- or detect policy failure and print a friendlier message

### B. `uv` not found

Symptoms:

- `where uv` fails
- install script exits early

Best next step:

- improve BAT scripts with clearer Windows install guidance
- optionally add `winget` hints directly in failure message

### C. Python mismatch

Symptoms:

- Python exists but unsupported version
- `uv sync` chooses unexpected interpreter

Best next step:

- check whether we should explicitly prefer `py -3.11`
- or document supported Windows Python more clearly

### D. Build/network/wheel issues

Symptoms:

- `uv sync` fails during package resolution or wheel build
- especially possible around `xgboost`, `numba`, `shap`

Best next step:

- identify exact package causing Windows pain
- consider documenting a known-good Python version

## Suggested Improvements For The Windows Codex Session

Priority order:

1. Verify that `scripts\install.bat` and `scripts\run_gui.bat` really work on a clean Windows machine.
2. Decide whether `bundle\make_bundle.bat` must become fully cmd-only.
3. If bundle build still depends on PowerShell, make that explicit in docs and errors.
4. Improve Windows-facing docs so users are told to prefer `cmd.exe` + BAT files.
5. If needed, add a single top-level Windows helper like `run_gui_windows.bat` for discoverability.

## Strong Candidate Fix

If Windows bundle build still fails in practice, the cleanest direction is likely:

- keep `install.bat` and `run_gui.bat`
- rewrite Windows bundle creation so it does not require PowerShell for control flow
- use Python or `uv run python -m zipfile` for archive creation instead of `Compress-Archive`

Reason:

- `Compress-Archive` is convenient, but it keeps PowerShell in the critical path
- that is exactly the path enterprise Windows machines tend to make fragile

## What To Return After Windows Testing

Please bring back:

- exact commands run
- exact error text
- whether testing was done in `cmd.exe` or PowerShell
- whether `scripts\run_gui.bat` worked
- whether `bundle\make_bundle.bat` worked
- if bundle built, whether `launch_gui.bat` inside the unzipped bundle worked

## Notes

- The repo currently has unrelated in-progress local UI/SHAP edits on macOS. Those are not part of this Windows handoff.
- Focus this Windows session only on install/run/bundle behavior.
