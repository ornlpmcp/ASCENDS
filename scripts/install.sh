#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[ASCENDS] Starting setup..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ASCENDS] ERROR: python3 is required but not found."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ASCENDS] 'uv' not found. Installing with pip (user scope)..."
  python3 -m pip install --user uv

  if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ASCENDS] ERROR: uv installation failed. Install uv manually and retry."
  echo "         https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[ASCENDS] Syncing dependencies..."
uv sync

echo "[ASCENDS] Setup complete."
echo "[ASCENDS] Run GUI with:"
echo "  ./scripts/run_gui.sh"
