$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DistDir = Join-Path $RootDir "dist"
$DateTag = Get-Date -Format "yyyyMMdd"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$OsTag = "windows"
$ArchTag = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()

$PyprojectPath = Join-Path $RootDir "pyproject.toml"
$PyprojectText = Get-Content -Raw -Path $PyprojectPath -Encoding UTF8
$VersionMatch = [regex]::Match($PyprojectText, '(?m)^version\s*=\s*"([^"]+)"')
$VersionTag = if ($VersionMatch.Success) { $VersionMatch.Groups[1].Value } else { "0.0.0" }

$BundleName = "ASCENDS-v$VersionTag-$DateTag-$OsTag"
$BundleRoot = Join-Path $DistDir $BundleName
$BundleApp  = Join-Path $BundleRoot "ASCENDS"
$ArchivePath = Join-Path $DistDir "$BundleName.zip"

Write-Host "[ASCENDS] Building portable bundle: $BundleName"
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path $BundleRoot) { Remove-Item -Recurse -Force $BundleRoot }
New-Item -ItemType Directory -Force -Path $BundleApp | Out-Null

# ── Copy source files ───────────────────────────────────────────────────────
Write-Host "[ASCENDS] Copying project files..."
$DirsToCopy = @("ascends", "templates", "static", "examples")
foreach ($d in $DirsToCopy) {
  $src = Join-Path $RootDir $d
  if (Test-Path $src) {
    Copy-Item -Recurse -Force -Path $src -Destination (Join-Path $BundleApp $d)
  }
}

$FilesToCopy = @(
  "ascends_server.py",
  "pyproject.toml",
  "uv.lock",
  "README.md",
  "quickstart.md",
  "LICENSE"
)
foreach ($f in $FilesToCopy) {
  $src = Join-Path $RootDir $f
  if (Test-Path $src) {
    Copy-Item -Force -Path $src -Destination (Join-Path $BundleApp $f)
  }
}

# ── Find and copy uv.exe into bundle root ───────────────────────────────────
Write-Host "[ASCENDS] Locating uv.exe..."
$UvExe = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $UvExe) {
  Write-Host "[ASCENDS] ERROR: uv not found on PATH. Install uv first: https://docs.astral.sh/uv/" -ForegroundColor Red
  exit 1
}
Write-Host "[ASCENDS] Bundling uv.exe from: $UvExe"
Copy-Item -Force -Path $UvExe -Destination (Join-Path $BundleRoot "uv.exe")

# ── Pre-build venv on this machine to speed up first launch ─────────────────
# The venv may have path issues on the target machine, but uv run will
# automatically detect and repair it before launching.
Write-Host "[ASCENDS] Pre-building virtual environment (speeds up first launch)..."
Push-Location $BundleApp
try {
  & (Join-Path $BundleRoot "uv.exe") sync --no-dev
} finally {
  Pop-Location
}

# ── Write bundle metadata ────────────────────────────────────────────────────
@"
name=$BundleName
version=$VersionTag
os=$OsTag
arch=$ArchTag
timestamp=$Ts
"@ | Set-Content -Path (Join-Path $BundleRoot "bundle-meta.txt") -Encoding UTF8

# ── Write launch_gui.bat (cmd.exe — primary launcher) ───────────────────────
@'
@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%ASCENDS"

echo [ASCENDS] Launching GUI at http://127.0.0.1:7777
echo [ASCENDS] Open your browser at: http://127.0.0.1:7777
echo.

"%ROOT%uv.exe" run ascends gui %*
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.bat") -Encoding ASCII

# ── Write launch_gui.ps1 (optional PowerShell launcher) ─────────────────────
@'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $Root "ASCENDS")
Write-Host "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
& (Join-Path $Root "uv.exe") run ascends gui @args
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.ps1") -Encoding UTF8

# ── Write launch_cli.bat ─────────────────────────────────────────────────────
@'
@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%ASCENDS"
"%ROOT%uv.exe" run ascends %*
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_cli.bat") -Encoding ASCII

# ── Write README-BUNDLE.txt ──────────────────────────────────────────────────
@"
ASCENDS Portable Bundle v$VersionTag
=====================================

No Python or uv installation required.

QUICK START
-----------
1. Unzip this archive anywhere on your machine.
2. Double-click launch_gui.bat  (or run it from cmd.exe)
3. Open your browser at: http://127.0.0.1:7777

NOTES
-----
- First launch may take 1-2 minutes while the environment is verified.
- Subsequent launches are fast.
- This bundle is Windows-only ($ArchTag).
- For CLI use: launch_cli.bat --help

"@ | Set-Content -Path (Join-Path $BundleRoot "README-BUNDLE.txt") -Encoding UTF8

# ── Create zip archive ───────────────────────────────────────────────────────
if (Test-Path $ArchivePath) {
  $n = 2
  while (Test-Path (Join-Path $DistDir "$BundleName-$n.zip")) { $n += 1 }
  $ArchivePath = Join-Path $DistDir "$BundleName-$n.zip"
}

Write-Host "[ASCENDS] Creating archive: $ArchivePath"
Compress-Archive -Path $BundleRoot -DestinationPath $ArchivePath -Force

Write-Host ""
Write-Host "[ASCENDS] Bundle complete!" -ForegroundColor Green
Write-Host "  Directory : $BundleRoot"
Write-Host "  Archive   : $ArchivePath"
