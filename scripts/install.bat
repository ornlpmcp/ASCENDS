@echo off
REM This file installs ASCENDS dependencies on Windows using cmd.exe only (no PowerShell policy dependency).
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo [ASCENDS] Starting setup...

where uv >nul 2>nul
if errorlevel 1 (
  echo [ASCENDS] ERROR: uv is required but not found.
  echo [ASCENDS]        Install uv first: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)

echo [ASCENDS] Syncing dependencies...
uv sync
if errorlevel 1 (
  echo [ASCENDS] ERROR: dependency sync failed.
  exit /b 1
)

echo [ASCENDS] Setup complete.
echo [ASCENDS] Run GUI with:
echo   scripts\run_gui.bat
exit /b 0

