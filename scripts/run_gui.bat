@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%.."
powershell -ExecutionPolicy Bypass -File "%ROOT%run_gui.ps1" %*
