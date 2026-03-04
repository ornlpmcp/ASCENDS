param(
  [ValidateSet("standard", "pro")]
  [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Policy: Windows install uses pro only.
if (-not [string]::IsNullOrWhiteSpace($Profile) -and $Profile -ne "pro") {
  Write-Host "[ASCENDS] Note: Windows install uses pro only. Overriding '$Profile' -> 'pro'."
}
$Profile = "pro"

Write-Host "[ASCENDS] Starting setup for profile=$Profile ..."

if (-not (Get-Command python -ErrorAction SilentlyContinue) -and -not (Get-Command python3 -ErrorAction SilentlyContinue)) {
  Write-Host "[ASCENDS] ERROR: python/python3 is required but not found." -ForegroundColor Red
  exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "[ASCENDS] ERROR: uv is required. Install uv first, then re-run install.ps1." -ForegroundColor Red
  Write-Host "          https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}

Write-Host "[ASCENDS] Syncing dependencies..."
if ($Profile -eq "pro") {
  uv sync --extra pro
} else {
  uv sync
}
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ASCENDS] ERROR: dependency sync failed." -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[ASCENDS] Setup complete."
Write-Host "[ASCENDS] Run GUI with:"
Write-Host "  ./run_gui.ps1"
