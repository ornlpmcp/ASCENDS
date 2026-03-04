#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/ascends" ]]; then
  echo "[ASCENDS] Environment not ready. Running install first..."
  ./scripts/install.sh
fi

echo "[ASCENDS] Launching GUI at http://127.0.0.1:7777"
exec .venv/bin/ascends gui "$@"
