#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-}" # standard | pro
OS_RAW="$(uname -s)"

# Policy: macOS/Windows-like environments use pro only. Linux can choose standard/pro.
if [[ "$OS_RAW" != "Linux" ]]; then
  if [[ -n "$PROFILE" && "$PROFILE" != "pro" ]]; then
    echo "[ASCENDS] Note: non-Linux install uses pro only. Overriding '$PROFILE' -> 'pro'."
  fi
  PROFILE="pro"
fi

if [[ -z "$PROFILE" ]]; then
  if [[ -t 0 ]]; then
    echo "[ASCENDS] Select installation profile:"
    echo "  1) standard (Recommended): lighter install, no xgboost/shap"
    echo "  2) pro: includes xgboost + shap"
    read -r -p "Enter 1 or 2 [1]: " _choice
    case "${_choice:-1}" in
      2) PROFILE="pro" ;;
      *) PROFILE="standard" ;;
    esac
  else
    PROFILE="standard"
  fi
fi

if [[ "$PROFILE" != "standard" && "$PROFILE" != "pro" ]]; then
  echo "[ASCENDS] ERROR: profile must be 'standard' or 'pro'."
  echo "Usage: ./install.sh [standard|pro]"
  exit 1
fi

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
if [[ "$PROFILE" == "pro" ]]; then
  uv sync --extra pro
else
  uv sync
fi

echo "[ASCENDS] Setup complete."
echo "[ASCENDS] Run GUI with:"
echo "  ./run_gui.sh"
