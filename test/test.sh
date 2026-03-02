#!/usr/bin/env bash
set -e  # Exit on first error
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/boston_rf"
CSV_TRAIN="$REPO_ROOT/examples/BostonHousing.csv"
CSV_TEST="$REPO_ROOT/examples/BostonHousing_test.csv"
CSV_CLASS="$REPO_ROOT/examples/iris.csv"

echo "=== [1/4] Running correlation ==="
uv run ascends correlation \
  --csv "$CSV_TRAIN" \
  --target medv \
  --task r \
  --metrics pearson,spearman,mi,dcor \
  --view wide \
  --out "$OUT_DIR/correlation.csv"

echo "=== [2/4] Training model ==="
uv run ascends train \
  --csv "$CSV_TRAIN" \
  --target medv \
  --task r \
  --out "$OUT_DIR"

echo "=== [3/4] Generating parity plots ==="
uv run ascends parity-plot "$OUT_DIR" \
  --scope combined \
  --out "$OUT_DIR"

echo "=== [4/4] Predicting ==="
uv run ascends predict "$OUT_DIR" \
  --csv "$CSV_TEST" \
  --out "$OUT_DIR/predict"

echo "=== [5/5] Classification (GUI backend smoke) ==="
uv run python "$SCRIPT_DIR/classification_gui_smoke.py" \
  --csv "$CSV_CLASS" \
  --target Name \
  --model rf \
  --test-size 0.2

echo "✅ Test pipeline completed successfully!"
