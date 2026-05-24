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
m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
print(m.group(1) if m else "0.0.0")
PY
)"

OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"
case "$OS_RAW" in
  Darwin) OS_TAG="macOS" ;;
  Linux)  OS_TAG="linux" ;;
  *)      OS_TAG="$(echo "$OS_RAW" | tr '[:upper:]' '[:lower:]')" ;;
esac
ARCH_TAG="$(echo "$ARCH_RAW" | tr '[:upper:]' '[:lower:]')"

BUNDLE_NAME="ASCENDS-v${VERSION_TAG}-${DATE_TAG}-${OS_TAG}-${ARCH_TAG}"
BUNDLE_ROOT="$DIST_DIR/$BUNDLE_NAME"
BUNDLE_APP="$BUNDLE_ROOT/ASCENDS"
ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.zip"

if [[ "$OS_TAG" == "linux" ]]; then
  echo "[ASCENDS] Linux self-contained bundles are not supported by this script." >&2
  echo "[ASCENDS] Use the developer workflow instead: git pull && uv sync && uv run ascends gui" >&2
  exit 1
fi

echo "[ASCENDS] Building portable bundle: $BUNDLE_NAME"
mkdir -p "$DIST_DIR"
rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_APP"

# ── Copy source files ─────────────────────────────────────────────────────────
echo "[ASCENDS] Copying project files..."
for d in ascends templates static examples; do
  if [[ -d "$ROOT_DIR/$d" ]]; then
    cp -R "$ROOT_DIR/$d" "$BUNDLE_APP/"
  fi
done

for f in ascends_server.py pyproject.toml uv.lock README.md quickstart.md LICENSE; do
  if [[ -f "$ROOT_DIR/$f" ]]; then
    cp "$ROOT_DIR/$f" "$BUNDLE_APP/"
  fi
done

# ── Find uv for build-time environment creation ───────────────────────────────
echo "[ASCENDS] Locating uv..."
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" && -x "/opt/homebrew/bin/uv" ]]; then
  UV_BIN="/opt/homebrew/bin/uv"
fi
if [[ -z "$UV_BIN" && -x "/usr/local/bin/uv" ]]; then
  UV_BIN="/usr/local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "[ASCENDS] ERROR: uv not found on PATH. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
echo "[ASCENDS] Using uv for build only: $UV_BIN"

BUILD_PYTHON="$(command -v python3.11 2>/dev/null || true)"
if [[ -z "$BUILD_PYTHON" && -x "/opt/homebrew/bin/python3.11" ]]; then
  BUILD_PYTHON="/opt/homebrew/bin/python3.11"
fi
if [[ -z "$BUILD_PYTHON" && -x "/usr/local/bin/python3.11" ]]; then
  BUILD_PYTHON="/usr/local/bin/python3.11"
fi

UV_SYNC_ARGS=(sync --no-dev --link-mode=copy)
if [[ -n "$BUILD_PYTHON" ]]; then
  echo "[ASCENDS] Using Python for build only: $BUILD_PYTHON"
  UV_SYNC_ARGS+=(--python "$BUILD_PYTHON" --python-preference only-system)
else
  echo "[ASCENDS] Python 3.11 not found; using uv-managed Python for this platform."
  UV_SYNC_ARGS+=(--python-preference managed)
fi

# ── Pre-build venv using build-machine uv ─────────────────────────────────────
echo "[ASCENDS] Pre-building virtual environment (speeds up first launch)..."
BUILD_UV_CACHE="$DIST_DIR/.uv-build-cache"
rm -rf "$BUILD_UV_CACHE"
mkdir -p "$BUILD_UV_CACHE"
pushd "$BUNDLE_APP" >/dev/null
UV_CACHE_DIR="$BUILD_UV_CACHE" UV_LINK_MODE=copy "$UV_BIN" "${UV_SYNC_ARGS[@]}"
popd >/dev/null
if [[ -f "$BUNDLE_APP/.venv/pyvenv.cfg" ]]; then
  sed -i.bak 's|^home = .*|home = ../../python/bin|' "$BUNDLE_APP/.venv/pyvenv.cfg"
  rm -f "$BUNDLE_APP/.venv/pyvenv.cfg.bak"
fi

# ── Bundle the Python runtime used by the venv ────────────────────────────────
echo "[ASCENDS] Locating Python runtime to bundle..."
PY_INFO="$("$BUNDLE_APP/.venv/bin/python" - <<'PY'
import sys
print(sys.base_prefix)
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PY_BASE="$(printf '%s\n' "$PY_INFO" | sed -n '1p')"
PY_VERSION="$(printf '%s\n' "$PY_INFO" | sed -n '2p')"
if [[ -z "$PY_BASE" || ! -d "$PY_BASE" ]]; then
  echo "[ASCENDS] ERROR: could not locate Python base prefix: $PY_BASE" >&2
  exit 1
fi
PY_BASE_REAL="$(cd "$PY_BASE" && pwd -P)"
echo "[ASCENDS] Bundling Python from: $PY_BASE_REAL"
cp -R "$PY_BASE_REAL" "$BUNDLE_ROOT/python"

if [[ "$OS_TAG" == "macOS" ]]; then
  PYTHON_EXE="$(find "$BUNDLE_ROOT/python/bin" -maxdepth 1 -type f -name 'python3*' | head -n 1)"
  PYTHON_DYLIB="$BUNDLE_ROOT/python/Python"
  if [[ -z "$PYTHON_EXE" || ! -x "$PYTHON_EXE" ]]; then
    echo "[ASCENDS] ERROR: bundled Python executable not found." >&2
    exit 1
  fi

  OLD_PYTHON_DYLIB="$(otool -L "$PYTHON_EXE" | awk '/Python\.framework\/Versions\/[0-9.]+\/Python/ {print $1; exit}')"
  if [[ -n "$OLD_PYTHON_DYLIB" && -f "$PYTHON_DYLIB" ]]; then
    echo "[ASCENDS] Rewriting bundled Python framework paths for portability..."
    install_name_tool -change "$OLD_PYTHON_DYLIB" "@executable_path/../Python" "$PYTHON_EXE"
    install_name_tool -id "@executable_path/../Python" "$PYTHON_DYLIB"
    codesign --force --sign - "$PYTHON_DYLIB" "$PYTHON_EXE" >/dev/null
  else
    echo "[ASCENDS] Bundled Python does not use Homebrew framework paths; no rewrite needed."
  fi
fi

# ── Bundle metadata ───────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/bundle-meta.txt" <<EOF
name=$BUNDLE_NAME
version=$VERSION_TAG
os=$OS_TAG
arch=$ARCH_TAG
python=$PY_VERSION
timestamp=$TS
EOF

# ── Launch scripts ────────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/launch_gui.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="\$ROOT_DIR/python/bin/python3"
if [[ ! -x "\$PYTHON_BIN" ]]; then
  PYTHON_BIN="\$(find "\$ROOT_DIR/python/bin" -maxdepth 1 -type f -name 'python3*' | head -n 1)"
fi
if [[ -z "\$PYTHON_BIN" || ! -x "\$PYTHON_BIN" ]]; then
  echo "[ASCENDS] ERROR: bundled Python not found." >&2
  exit 1
fi
export PYTHONPATH="\$ROOT_DIR/ASCENDS/.venv/lib/python${PY_VERSION}/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONHOME="\$ROOT_DIR/python"
export NUMBA_CACHE_DIR="\${TMPDIR:-/tmp}/ascends_numba_cache"
export MPLCONFIGDIR="\${TMPDIR:-/tmp}/ascends_matplotlib_cache"
mkdir -p "\$NUMBA_CACHE_DIR"
mkdir -p "\$MPLCONFIGDIR"
cd "\$ROOT_DIR/ASCENDS"
echo "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
echo "[ASCENDS] Browser should open automatically at http://127.0.0.1:7777"
echo "[ASCENDS] First launch may take 1-2 minutes (compiling math libraries)."
echo
exec "\$PYTHON_BIN" -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" gui --no-reload --open-browser "\$@"
EOF

cat > "$BUNDLE_ROOT/launch_cli.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="\$ROOT_DIR/python/bin/python3"
if [[ ! -x "\$PYTHON_BIN" ]]; then
  PYTHON_BIN="\$(find "\$ROOT_DIR/python/bin" -maxdepth 1 -type f -name 'python3*' | head -n 1)"
fi
if [[ -z "\$PYTHON_BIN" || ! -x "\$PYTHON_BIN" ]]; then
  echo "[ASCENDS] ERROR: bundled Python not found." >&2
  exit 1
fi
export PYTHONPATH="\$ROOT_DIR/ASCENDS/.venv/lib/python${PY_VERSION}/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONHOME="\$ROOT_DIR/python"
export NUMBA_CACHE_DIR="\${TMPDIR:-/tmp}/ascends_numba_cache"
export MPLCONFIGDIR="\${TMPDIR:-/tmp}/ascends_matplotlib_cache"
mkdir -p "\$NUMBA_CACHE_DIR"
mkdir -p "\$MPLCONFIGDIR"
cd "\$ROOT_DIR/ASCENDS"
exec "\$PYTHON_BIN" -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" "\$@"
EOF

chmod +x "$BUNDLE_ROOT/launch_gui.sh" "$BUNDLE_ROOT/launch_cli.sh"
if [[ "$OS_TAG" == "macOS" ]]; then
  cp "$BUNDLE_ROOT/launch_gui.sh" "$BUNDLE_ROOT/launch_gui.command"
  chmod +x "$BUNDLE_ROOT/launch_gui.command"
fi

# ── README-BUNDLE.txt ─────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/README-BUNDLE.txt" <<EOF
ASCENDS Portable Bundle v${VERSION_TAG}
=====================================

No Python or uv installation required.

QUICK START
-----------
1. Unzip this archive anywhere on your machine.
2. macOS: double-click launch_gui.command
   Linux: run ./launch_gui.sh
3. Your browser should open automatically at http://127.0.0.1:7777

NOTES
-----
- First launch may take 1-2 minutes while the environment is verified.
- Subsequent launches are fast.
- This bundle is ${OS_TAG} (${ARCH_TAG}).
- For CLI use: ./launch_cli.sh --help
- macOS may show a first-run security warning for downloaded unsigned scripts.
EOF

# ── Create archive ────────────────────────────────────────────────────────────
if [[ -f "$ARCHIVE_PATH" ]]; then
  N=2
  while [[ -f "$DIST_DIR/${BUNDLE_NAME}-${N}.zip" ]]; do
    N=$((N + 1))
  done
  ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}-${N}.zip"
fi

echo "[ASCENDS] Creating archive: $ARCHIVE_PATH"
python3 - "$DIST_DIR" "$BUNDLE_NAME" "$ARCHIVE_PATH" <<'PY'
import sys
import zipfile
from pathlib import Path

dist_dir = Path(sys.argv[1])
bundle_name = sys.argv[2]
archive_path = Path(sys.argv[3])
root = dist_dir / bundle_name

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob("*"):
        rel = path.relative_to(dist_dir)
        info = zipfile.ZipInfo(str(rel))
        stat_result = path.lstat()
        mode = stat_result.st_mode
        info.external_attr = (mode & 0xFFFF) << 16
        if path.is_symlink():
            info.create_system = 3
            zf.writestr(info, str(path.readlink()).encode("utf-8"))
        elif path.is_dir():
            info.filename += "/"
            zf.writestr(info, b"")
        else:
            with path.open("rb") as f:
                zf.writestr(info, f.read())
PY

echo ""
echo "[ASCENDS] Bundle complete!"
echo "  Directory : $BUNDLE_ROOT"
echo "  Archive   : $ARCHIVE_PATH"
