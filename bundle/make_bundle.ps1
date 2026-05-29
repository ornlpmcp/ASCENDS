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

$BundleName = "ASCENDS-v$VersionTag-$DateTag-$OsTag-$ArchTag"
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

# ── Find uv.exe for build-time environment creation ─────────────────────────
Write-Host "[ASCENDS] Locating uv.exe..."
$UvExe = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $UvExe) {
  Write-Host "[ASCENDS] ERROR: uv not found on PATH. Install uv first: https://docs.astral.sh/uv/" -ForegroundColor Red
  exit 1
}
Write-Host "[ASCENDS] Using uv.exe for build only: $UvExe"

# ── Pre-build venv (installs all packages into .venv\Lib\site-packages) ──────
Write-Host "[ASCENDS] Pre-building virtual environment..."
Push-Location $BundleApp
try {
  & $UvExe sync --no-dev
} finally {
  Pop-Location
}

# ── Bundle the Python distribution so the launcher is fully self-contained ───
# pyvenv.cfg bakes in an absolute 'home' path that won't exist on other PCs.
# Fix: copy the Python distribution alongside the bundle and run python.exe
# directly, bypassing the venv stub entirely.
Write-Host "[ASCENDS] Locating Python distribution to bundle..."
$PyvenvCfg  = Join-Path $BundleApp ".venv\pyvenv.cfg"
$PythonHome = (Get-Content $PyvenvCfg |
               Where-Object { $_ -match '^home\s*=' } |
               ForEach-Object { ($_ -split '=', 2)[1].Trim() })
if (-not $PythonHome -or -not (Test-Path $PythonHome)) {
  Write-Host "[ASCENDS] ERROR: could not read Python home from pyvenv.cfg." -ForegroundColor Red
  Write-Host "  home = $PythonHome"
  exit 1
}
Write-Host "[ASCENDS] Bundling Python from: $PythonHome"
Copy-Item -Recurse -Force -Path $PythonHome -Destination (Join-Path $BundleRoot "python")

# ── Write bundle metadata ────────────────────────────────────────────────────
@"
name=$BundleName
version=$VersionTag
os=$OsTag
arch=$ArchTag
timestamp=$Ts
"@ | Set-Content -Path (Join-Path $BundleRoot "bundle-meta.txt") -Encoding UTF8

# ── Write launch_gui.bat (cmd.exe — primary launcher) ───────────────────────
# Uses bundled python\python.exe directly (bypasses venv stub + pyvenv.cfg).
# PYTHONPATH points to the pre-built site-packages so all deps are found.
@'
@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
REM Isolate from user's Python environment (v0.8.2 fix)
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONUSERBASE="
set "VIRTUAL_ENV="
set "CONDA_DEFAULT_ENV="
set "CONDA_PREFIX="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%ROOT%ASCENDS\.venv\Lib\site-packages"
set "PYTHONUTF8=1"
set "NUMBA_CACHE_DIR=%TEMP%\nc"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
cd /d "%ROOT%ASCENDS"

echo [ASCENDS] Launching GUI at http://127.0.0.1:7777
echo [ASCENDS] Browser should open automatically at http://127.0.0.1:7777
echo [ASCENDS] First launch may take 1-2 minutes (compiling math libraries).
echo.

"%ROOT%python\python.exe" -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" gui --no-reload --open-browser %*
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.bat") -Encoding ASCII

# ── Write launch_gui.ps1 (optional PowerShell launcher) ─────────────────────
@'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
# Isolate from user's Python environment (v0.8.2 fix)
$env:PYTHONPATH = $null
$env:PYTHONHOME = $null
$env:PYTHONSTARTUP = $null
$env:PYTHONUSERBASE = $null
$env:VIRTUAL_ENV = $null
$env:CONDA_DEFAULT_ENV = $null
$env:CONDA_PREFIX = $null
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = Join-Path $Root "ASCENDS\.venv\Lib\site-packages"
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP "nc"
if (-not (Test-Path $env:NUMBA_CACHE_DIR)) { New-Item -ItemType Directory -Force -Path $env:NUMBA_CACHE_DIR | Out-Null }
Set-Location (Join-Path $Root "ASCENDS")
Write-Host "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
$PyExe = Join-Path $Root "python\python.exe"
& $PyExe -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" "gui" "--no-reload" "--open-browser" @args
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.ps1") -Encoding UTF8

# ── Write launch_cli.bat ─────────────────────────────────────────────────────
@'
@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
REM Isolate from user's Python environment (v0.8.2 fix)
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONUSERBASE="
set "VIRTUAL_ENV="
set "CONDA_DEFAULT_ENV="
set "CONDA_PREFIX="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%ROOT%ASCENDS\.venv\Lib\site-packages"
set "PYTHONUTF8=1"
cd /d "%ROOT%ASCENDS"
"%ROOT%python\python.exe" -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" %*
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
3. Your browser should open automatically at http://127.0.0.1:7777

NOTES
-----
- No internet connection required — Python and all packages are included.
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
$ArchiveParent = Split-Path -Parent $ArchivePath
$ArchiveLeaf = Split-Path -Leaf $ArchivePath
$BundleParent = Split-Path -Parent $BundleRoot
$BundleLeaf = Split-Path -Leaf $BundleRoot
$TarExe = (Get-Command tar.exe -ErrorAction SilentlyContinue).Source
if ($TarExe) {
  Write-Host "[ASCENDS] Using tar.exe for faster zip creation..."
  Push-Location $BundleParent
  try {
    & $TarExe -a -cf (Join-Path $ArchiveParent $ArchiveLeaf) $BundleLeaf
  } finally {
    Pop-Location
  }
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ArchivePath) -or ((Get-Item $ArchivePath).Length -le 0)) {
    Write-Host "[ASCENDS] tar.exe failed; falling back to Compress-Archive..." -ForegroundColor Yellow
    Compress-Archive -Path $BundleRoot -DestinationPath $ArchivePath -Force
  }
} else {
  Write-Host "[ASCENDS] tar.exe not found; falling back to Compress-Archive..." -ForegroundColor Yellow
  Compress-Archive -Path $BundleRoot -DestinationPath $ArchivePath -Force
}

Write-Host ""
Write-Host "[ASCENDS] Bundle complete!" -ForegroundColor Green
Write-Host "  Directory : $BundleRoot"
Write-Host "  Archive   : $ArchivePath"
