$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$AscendsExe = ".venv\Scripts\ascends.exe"
if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] Environment not ready. Running install first..."
  & "$RootDir\scripts\install.ps1"
}

if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] ERROR: ASCENDS executable not found after install." -ForegroundColor Red
  exit 1
}

Write-Host "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
& $AscendsExe gui @args
