#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
TS="$(date +%Y%m%d_%H%M%S)"
PROFILE="${1:-pro}" # standard | pro

OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"
case "$OS_RAW" in
  Darwin) OS_TAG="macos" ;;
  Linux) OS_TAG="linux" ;;
  *) OS_TAG="$(echo "$OS_RAW" | tr '[:upper:]' '[:lower:]')" ;;
esac
ARCH_TAG="$(echo "$ARCH_RAW" | tr '[:upper:]' '[:lower:]')"

if [[ "$PROFILE" != "standard" && "$PROFILE" != "pro" ]]; then
  echo "[ASCENDS] ERROR: profile must be 'standard' or 'pro'."
  echo "Usage: bash ./bundle/make_bundle.sh [standard|pro]"
  exit 1
fi

BUNDLE_NAME="ASCENDS-bundle-${PROFILE}-${OS_TAG}-${ARCH_TAG}-${TS}"
BUNDLE_ROOT="$DIST_DIR/$BUNDLE_NAME"
BUNDLE_APP="$BUNDLE_ROOT/ASCENDS"
ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.tar.gz"

echo "[ASCENDS] Preparing portable bundle (profile=$PROFILE)..."
mkdir -p "$DIST_DIR"
rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_APP"

if [[ ! -x "$ROOT_DIR/.venv/bin/ascends" ]]; then
  echo "[ASCENDS] Local environment not ready, running install.sh first..."
  bash "$ROOT_DIR/install.sh"
fi

echo "[ASCENDS] Copying project files..."
cp -R "$ROOT_DIR/ascends" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/templates" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/static" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/examples" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/test" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/.venv" "$BUNDLE_APP/"

cp "$ROOT_DIR/ascends_server.py" "$BUNDLE_APP/"
cp "$ROOT_DIR/pyproject.toml" "$BUNDLE_APP/"
cp "$ROOT_DIR/uv.lock" "$BUNDLE_APP/"
cp "$ROOT_DIR/README.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/README.dev.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/quickstart.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/LICENSE" "$BUNDLE_APP/"
cp "$ROOT_DIR/CHANGELOG.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/TODO.md" "$BUNDLE_APP/"

# Bundle metadata
cat > "$BUNDLE_ROOT/bundle-meta.txt" <<EOF
name=$BUNDLE_NAME
profile=$PROFILE
os=$OS_TAG
arch=$ARCH_TAG
timestamp=$TS
EOF

cat > "$BUNDLE_ROOT/launch_gui.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/ASCENDS"

if [[ ! -x ".venv/bin/ascends" ]]; then
  echo "[ASCENDS] ERROR: bundled environment is missing (.venv/bin/ascends)."
  exit 1
fi

echo "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
exec .venv/bin/ascends gui "$@"
EOF

cat > "$BUNDLE_ROOT/launch_cli.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/ASCENDS"

if [[ ! -x ".venv/bin/ascends" ]]; then
  echo "[ASCENDS] ERROR: bundled environment is missing (.venv/bin/ascends)."
  exit 1
fi

exec .venv/bin/ascends "$@"
EOF

cat > "$BUNDLE_ROOT/launch_gui.ps1" <<'EOF'
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
EOF

cat > "$BUNDLE_ROOT/launch_cli.ps1" <<'EOF'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $Root "ASCENDS")

$AscendsExe = ".venv\Scripts\ascends.exe"
if (-not (Test-Path $AscendsExe)) {
  Write-Host "[ASCENDS] ERROR: bundled environment is missing (.venv\Scripts\ascends.exe)." -ForegroundColor Red
  exit 1
}

& $AscendsExe @args
EOF

cat > "$BUNDLE_ROOT/launch_gui.bat" <<'EOF'
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
EOF

cat > "$BUNDLE_ROOT/README-BUNDLE.txt" <<'EOF'
ASCENDS Portable Bundle
=======================

1) Unpack this bundle on the same OS/architecture family it was built on.
2) Launch GUI:
   macOS/Linux:
   ./launch_gui.sh
   Windows (PowerShell):
   .\launch_gui.ps1
   Windows (cmd):
   launch_gui.bat
3) Open in browser:
   http://127.0.0.1:7777

Notes:
- This bundle is OS/architecture specific.
- It includes a prebuilt Python environment (.venv).
- For CLI use (macOS/Linux):
  ./launch_cli.sh --help
- For CLI use (Windows PowerShell):
  .\launch_cli.ps1 --help
EOF

chmod +x "$BUNDLE_ROOT/launch_gui.sh" "$BUNDLE_ROOT/launch_cli.sh"

echo "[ASCENDS] Creating archive: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$DIST_DIR" "$BUNDLE_NAME"

echo "[ASCENDS] Bundle complete."
echo "  Directory: $BUNDLE_ROOT"
echo "  Archive:   $ARCHIVE_PATH"
