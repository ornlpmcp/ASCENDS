param(
  [ValidateSet("standard", "pro")]
  [string]$Profile = "pro"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Policy: Windows run uses pro only.
if ($Profile -ne "pro") {
  Write-Host "[ASCENDS] Note: Windows run uses pro only. Overriding '$Profile' -> 'pro'."
  $Profile = "pro"
}

$AscendsExe = ".venv\Scripts\ascends.exe"
if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] Environment not ready. Running install first..."
  & "$RootDir\install.ps1" -Profile $Profile
}

if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] ERROR: ASCENDS executable not found after install." -ForegroundColor Red
  exit 1
}

Write-Host "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
& $AscendsExe gui @args
