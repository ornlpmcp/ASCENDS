Write-Host "=== [1/3] Running correlation ==="
uv run ascends correlation `
  --csv examples/BostonHousing.csv `
  --target medv `
  --task r `
  --metrics pearson,spearman,mi,dcor `
  --view wide `
  --out runs/boston_rf/correlation.csv

Write-Host "=== [2/3] Training model ==="
uv run ascends train `
  --csv examples/BostonHousing.csv `
  --target medv `
  --task r `
  --out runs/boston_rf

Write-Host "=== [3/3] Predicting ==="
uv run ascends predict runs/boston_rf `
  --csv examples/BostonHousing_test.csv `
  --out runs/predict

Write-Host "✅ Test completed successfully!"

