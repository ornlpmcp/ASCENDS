#!/usr/bin/env bash
set -e  # Exit on first error
set -o pipefail



echo "=== [1/4] Running correlation ==="
uv run ascends correlation \
  --csv ../examples/BostonHousing.csv \
  --target medv \
  --task r \
  --metrics pearson,spearman,mi,dcor \
  --view wide \
  --out ./boston_rf/correlation.csv

echo "=== [2/4] Training model ==="
uv run ascends train \
  --csv ../examples/BostonHousing.csv \
  --target medv \
  --task r \
  --out ./boston_rf

echo "=== [3/4] Generating parity plots ==="
uv run ascends parity-plot ./boston_rf \
  --scope combined \
  --out ./boston_rf

echo "=== [4/4] Predicting ==="
uv run ascends predict ./boston_rf \
  --csv ../examples/BostonHousing_test.csv \
  --out ./boston_rf/predict

echo "✅ Test pipeline completed successfully!"

