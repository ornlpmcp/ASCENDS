# test.ps1
# Exit immediately if any command fails
$ErrorActionPreference = "Stop"

Write-Host "=== [1/5] Running correlation ==="
uv run ascends correlation `
  --csv ../examples/BostonHousing.csv `
  --target medv `
  --task r `
  --metrics pearson,spearman,mi,dcor `
  --view wide `
  --out ./boston_rf/correlation.csv

Write-Host "=== [2/5] Training model ==="
uv run ascends train `
  --csv ../examples/BostonHousing.csv `
  --target medv `
  --task r `
  --out ./boston_rf

Write-Host "=== [3/5] Generating parity plots ==="
uv run ascends parity-plot ./boston_rf `
  --scope combined `
  --out ./boston_rf

Write-Host "=== [4/5] Predicting ==="
uv run ascends predict ./boston_rf `
  --csv ../examples/BostonHousing_test.csv `
  --out ./boston_rf/predict

Write-Host "=== [5/5] Classification (GUI backend smoke) ==="
uv run python ./classification_gui_smoke.py `
  --csv ../examples/iris.csv `
  --target Name `
  --model rf `
  --test-size 0.2

Write-Host "✅ Test pipeline completed successfully!"
