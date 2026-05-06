@echo off
REM This file builds the Windows ASCENDS bundle from cmd.exe.
REM It reads the PowerShell build script as text to avoid unsigned .ps1 execution-policy blocks.
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$script = Get-Content -LiteralPath '%ROOT%\bundle\make_bundle.ps1' -Raw; $block = [scriptblock]::Create($script); & $block"
exit /b %errorlevel%
