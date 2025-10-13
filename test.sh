#!/usr/bin/env bash
set -e  # Exit on first error
set -o pipefail

echo "=== [1/4] Running correlation ==="
uv run ascends correlation \
  --csv examples/BostonHousing.csv \
  --target medv \
  --task r \
  --metrics pearson,spearman,mi,dcor \
  --view wide \
  --out runs/boston_rf/correlation.csv

echo "=== [2/4] Training model ==="
uv run ascends train \
  --csv examples/BostonHousing.csv \
  --target medv \
  --task r \
  --out runs/boston_rf

echo "=== [3/4] Generating parity plots ==="
uv run ascends parity-plot runs/boston_rf \
  --scope combined \
  --out runs/boston_rf

echo "=== [4/4] Predicting ==="
uv run ascends predict runs/boston_rf \
  --csv examples/BostonHousing_test.csv \
  --out runs/boston_rf/predict

echo "✅ Test pipeline completed successfully!"

