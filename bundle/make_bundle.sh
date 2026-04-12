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

BUNDLE_NAME="ASCENDS-v${VERSION_TAG}-${DATE_TAG}-${OS_TAG}"
BUNDLE_ROOT="$DIST_DIR/$BUNDLE_NAME"
BUNDLE_APP="$BUNDLE_ROOT/ASCENDS"
ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.tar.gz"

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

# ── Find and copy uv binary into bundle root ──────────────────────────────────
echo "[ASCENDS] Locating uv..."
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "[ASCENDS] ERROR: uv not found on PATH. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
echo "[ASCENDS] Bundling uv from: $UV_BIN"
cp "$UV_BIN" "$BUNDLE_ROOT/uv"
chmod +x "$BUNDLE_ROOT/uv"

# ── Pre-build venv using bundled uv ───────────────────────────────────────────
echo "[ASCENDS] Pre-building virtual environment (speeds up first launch)..."
pushd "$BUNDLE_APP" >/dev/null
"$BUNDLE_ROOT/uv" sync --no-dev
popd >/dev/null

# ── Bundle metadata ───────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/bundle-meta.txt" <<EOF
name=$BUNDLE_NAME
version=$VERSION_TAG
os=$OS_TAG
arch=$ARCH_TAG
timestamp=$TS
EOF

# ── Launch scripts ────────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/launch_gui.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/ASCENDS"
echo "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
echo "[ASCENDS] Open your browser at: http://127.0.0.1:7777"
exec "$ROOT_DIR/uv" run ascends gui "$@"
EOF

cat > "$BUNDLE_ROOT/launch_cli.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/ASCENDS"
exec "$ROOT_DIR/uv" run ascends "$@"
EOF

chmod +x "$BUNDLE_ROOT/launch_gui.sh" "$BUNDLE_ROOT/launch_cli.sh"

# ── README-BUNDLE.txt ─────────────────────────────────────────────────────────
cat > "$BUNDLE_ROOT/README-BUNDLE.txt" <<EOF
ASCENDS Portable Bundle v${VERSION_TAG}
=====================================

No Python or uv installation required.

QUICK START
-----------
1. Unpack this archive anywhere on your machine.
2. Run: ./launch_gui.sh
3. Open your browser at: http://127.0.0.1:7777

NOTES
-----
- First launch may take 1-2 minutes while the environment is verified.
- Subsequent launches are fast.
- This bundle is ${OS_TAG} (${ARCH_TAG}).
- For CLI use: ./launch_cli.sh --help
EOF

# ── Create archive ────────────────────────────────────────────────────────────
if [[ -f "$ARCHIVE_PATH" ]]; then
  N=2
  while [[ -f "$DIST_DIR/${BUNDLE_NAME}-${N}.tar.gz" ]]; do
    N=$((N + 1))
  done
  ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}-${N}.tar.gz"
fi

echo "[ASCENDS] Creating archive: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$DIST_DIR" "$BUNDLE_NAME"

echo ""
echo "[ASCENDS] Bundle complete!"
echo "  Directory : $BUNDLE_ROOT"
echo "  Archive   : $ARCHIVE_PATH"
