@echo off
REM This file builds the Windows ASCENDS bundle from cmd.exe by invoking PowerShell with process-scope bypass.
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\bundle\make_bundle.ps1" %*
exit /b %errorlevel%

