@echo off
REM This file launches ASCENDS GUI on Windows using cmd.exe only (no PowerShell policy dependency).
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist ".venv\Scripts\ascends.exe" (
  echo [ASCENDS] Environment not ready. Running install first...
  call "%ROOT%\scripts\install.bat"
  if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\ascends.exe" (
  echo [ASCENDS] ERROR: ASCENDS executable not found after install.
  exit /b 1
)

echo [ASCENDS] Launching GUI at http://127.0.0.1:7777
".venv\Scripts\ascends.exe" gui %*
exit /b %errorlevel%
