$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DistDir = Join-Path $RootDir "dist"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$DateTag = Get-Date -Format "yyyyMMdd"
$OsTag = "windows"
$ArchTag = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()

$PyprojectPath = Join-Path $RootDir "pyproject.toml"
$PyprojectText = Get-Content -Raw -Path $PyprojectPath -Encoding UTF8
$VersionMatch = [regex]::Match($PyprojectText, '(?m)^version\s*=\s*"([^"]+)"')
$VersionTag = if ($VersionMatch.Success) { $VersionMatch.Groups[1].Value } else { "0.0.0" }

$BundleName = "ASCENDS-v$VersionTag-$DateTag-$OsTag"
$BundleRoot = Join-Path $DistDir $BundleName
$BundleApp = Join-Path $BundleRoot "ASCENDS"
$ArchiveExt = "zip"
$ArchivePath = Join-Path $DistDir "$BundleName.$ArchiveExt"

Write-Host "[ASCENDS] Preparing portable bundle..."
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path $BundleRoot) { Remove-Item -Recurse -Force $BundleRoot }
New-Item -ItemType Directory -Force -Path $BundleApp | Out-Null

Write-Host "[ASCENDS] Copying project files..."
$DirsToCopy = @("ascends", "templates", "static", "examples", "test")
foreach ($d in $DirsToCopy) {
  Copy-Item -Recurse -Force -Path (Join-Path $RootDir $d) -Destination (Join-Path $BundleApp $d)
}

$FilesToCopy = @(
  "ascends_server.py",
  "pyproject.toml",
  "uv.lock",
  "README.md",
  "README.dev.md",
  "quickstart.md",
  "LICENSE",
  "CHANGELOG.md",
  "TODO.md"
)
foreach ($f in $FilesToCopy) {
  Copy-Item -Force -Path (Join-Path $RootDir $f) -Destination (Join-Path $BundleApp $f)
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "[ASCENDS] ERROR: uv is required to create bundle environments." -ForegroundColor Red
  exit 1
}

Write-Host "[ASCENDS] Building bundled virtual environment..."
Push-Location $BundleApp
try {
  uv sync --no-dev
} finally {
  Pop-Location
}

@"
name=$BundleName
os=$OsTag
arch=$ArchTag
timestamp=$Ts
"@ | Set-Content -Path (Join-Path $BundleRoot "bundle-meta.txt") -Encoding UTF8

@'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $Root "ASCENDS")

$AscendsExe = ".venv\Scripts\ascends.exe"
if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] ERROR: bundled environment is missing (.venv\Scripts\ascends.exe)." -ForegroundColor Red
  exit 1
}

Write-Host "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
& $AscendsExe gui @args
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.ps1") -Encoding UTF8

@'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $Root "ASCENDS")

$AscendsExe = ".venv\Scripts\ascends.exe"
if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] ERROR: bundled environment is missing (.venv\Scripts\ascends.exe)." -ForegroundColor Red
  exit 1
}

& $AscendsExe @args
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_cli.ps1") -Encoding UTF8

@'
@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%ASCENDS"
if not exist ".venv\Scripts\ascends.exe" (
  echo [ASCENDS] ERROR: bundled environment is missing (.venv\Scripts\ascends.exe).
  exit /b 1
)
echo [ASCENDS] Launching GUI at http://127.0.0.1:7777
".venv\Scripts\ascends.exe" gui %*
'@ | Set-Content -Path (Join-Path $BundleRoot "launch_gui.bat") -Encoding ASCII

@'
ASCENDS Portable Bundle
=======================

1) Unpack this bundle on a compatible Windows machine.
2) Launch GUI:
   PowerShell:
   .\launch_gui.ps1
   cmd:
   launch_gui.bat
3) Open in browser:
   http://127.0.0.1:7777

Notes:
- This bundle is OS/architecture specific.
- It includes a prebuilt Python environment (.venv).
- For CLI use:
  .\launch_cli.ps1 --help
'@ | Set-Content -Path (Join-Path $BundleRoot "README-BUNDLE.txt") -Encoding UTF8

if (Test-Path $ArchivePath) {
  $n = 2
  while (Test-Path (Join-Path $DistDir "$BundleName-$n.$ArchiveExt")) {
    $n += 1
  }
  $ArchivePath = Join-Path $DistDir "$BundleName-$n.$ArchiveExt"
}

Write-Host "[ASCENDS] Creating archive: $ArchivePath"
Compress-Archive -Path $BundleRoot -DestinationPath $ArchivePath -Force

Write-Host "[ASCENDS] Bundle complete."
Write-Host "  Directory: $BundleRoot"
Write-Host "  Archive:   $ArchivePath"
