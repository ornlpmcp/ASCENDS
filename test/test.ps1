# test.ps1
# Exit immediately if any command fails
$ErrorActionPreference = "Stop"

Write-Host "=== [1/4] Running correlation ==="
uv run ascends correlation `
  --csv ../examples/BostonHousing.csv `
  --target medv `
  --task r `
  --metrics pearson,spearman,mi,dcor `
  --view wide `
  --out ./boston_rf/correlation.csv

Write-Host "=== [2/4] Training model ==="
uv run ascends train `
  --csv ../examples/BostonHousing.csv `
  --target medv `
  --task r `
  --out ./boston_rf

Write-Host "=== [3/4] Generating parity plots ==="
uv run ascends parity-plot ./boston_rf `
  --scope combined `
  --out ./boston_rf

Write-Host "=== [4/4] Predicting ==="
uv run ascends predict ./boston_rf `
  --csv ../examples/BostonHousing_test.csv `
  --out ./boston_rf/predict

Write-Host "✅ Test pipeline completed successfully!"

