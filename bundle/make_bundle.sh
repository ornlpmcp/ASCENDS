#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
TS="$(date +%Y%m%d_%H%M%S)"
DATE_TAG="$(date +%Y%m%d)"
VERSION_TAG="$(python3 - <<PY
import re
from pathlib import Path
p = Path(r"$ROOT_DIR/pyproject.toml")
text = p.read_text(encoding="utf-8")
m = re.search(r'^version\\s*=\\s*\"([^\"]+)\"', text, flags=re.MULTILINE)
print(m.group(1) if m else "0.0.0")
PY
)"

OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"
case "$OS_RAW" in
  Darwin) OS_TAG="macOS" ;;
  Linux) OS_TAG="linux" ;;
  *) OS_TAG="$(echo "$OS_RAW" | tr '[:upper:]' '[:lower:]')" ;;
esac
ARCH_TAG="$(echo "$ARCH_RAW" | tr '[:upper:]' '[:lower:]')"

BUNDLE_NAME="ASCENDS-v${VERSION_TAG}-${DATE_TAG}-${OS_TAG}"
BUNDLE_ROOT="$DIST_DIR/$BUNDLE_NAME"
BUNDLE_APP="$BUNDLE_ROOT/ASCENDS"
if [[ "$OS_TAG" == "windows" ]]; then
  ARCHIVE_EXT="zip"
else
  ARCHIVE_EXT="tar.gz"
fi
ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.${ARCHIVE_EXT}"

echo "[ASCENDS] Preparing portable bundle..."
mkdir -p "$DIST_DIR"
rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_APP"

echo "[ASCENDS] Copying project files..."
cp -R "$ROOT_DIR/ascends" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/templates" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/static" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/examples" "$BUNDLE_APP/"
cp -R "$ROOT_DIR/test" "$BUNDLE_APP/"

cp "$ROOT_DIR/ascends_server.py" "$BUNDLE_APP/"
cp "$ROOT_DIR/pyproject.toml" "$BUNDLE_APP/"
cp "$ROOT_DIR/uv.lock" "$BUNDLE_APP/"
cp "$ROOT_DIR/README.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/README.dev.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/quickstart.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/LICENSE" "$BUNDLE_APP/"
cp "$ROOT_DIR/CHANGELOG.md" "$BUNDLE_APP/"
cp "$ROOT_DIR/TODO.md" "$BUNDLE_APP/"

echo "[ASCENDS] Building bundled virtual environment..."
if ! command -v uv >/dev/null 2>&1; then
  echo "[ASCENDS] ERROR: uv is required to create bundle environments."
  exit 1
fi

pushd "$BUNDLE_APP" >/dev/null
uv sync --no-dev
popd >/dev/null

# Bundle metadata
cat > "$BUNDLE_ROOT/bundle-meta.txt" <<EOF
name=$BUNDLE_NAME
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

if [[ -f "$ARCHIVE_PATH" ]]; then
  N=2
  while [[ -f "$DIST_DIR/${BUNDLE_NAME}-${N}.${ARCHIVE_EXT}" ]]; do
    N=$((N + 1))
  done
  ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}-${N}.${ARCHIVE_EXT}"
fi

echo "[ASCENDS] Creating archive: $ARCHIVE_PATH"
if [[ "$ARCHIVE_EXT" == "zip" ]]; then
  python3 - <<PY
import os
import zipfile

dist_dir = r"$DIST_DIR"
bundle_name = r"$BUNDLE_NAME"
archive_path = r"$ARCHIVE_PATH"
bundle_root = os.path.join(dist_dir, bundle_name)

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(bundle_root):
        for fn in files:
            fp = os.path.join(root, fn)
            arcname = os.path.relpath(fp, dist_dir)
            zf.write(fp, arcname)
PY
else
  tar -czf "$ARCHIVE_PATH" -C "$DIST_DIR" "$BUNDLE_NAME"
fi

echo "[ASCENDS] Bundle complete."
echo "  Directory: $BUNDLE_ROOT"
echo "  Archive:   $ARCHIVE_PATH"
