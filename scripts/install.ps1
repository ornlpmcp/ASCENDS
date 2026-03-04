$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

Write-Host "[ASCENDS] Starting setup..."

if (-not (Get-Command python -ErrorAction SilentlyContinue) -and -not (Get-Command python3 -ErrorAction SilentlyContinue)) {
  Write-Host "[ASCENDS] ERROR: python/python3 is required but not found." -ForegroundColor Red
  exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "[ASCENDS] ERROR: uv is required. Install uv first, then re-run scripts/install.ps1." -ForegroundColor Red
  Write-Host "          https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}

Write-Host "[ASCENDS] Syncing dependencies..."
uv sync
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ASCENDS] ERROR: dependency sync failed." -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[ASCENDS] Setup complete."
Write-Host "[ASCENDS] Run GUI with:"
Write-Host "  ./scripts/run_gui.ps1"
