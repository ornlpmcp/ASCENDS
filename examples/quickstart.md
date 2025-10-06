# Quickstart

```bash
uv run ascends correlation --csv examples/sample.csv --target y --task r
uv run ascends train --csv examples/sample.csv --target y --task r --model xgb --tune quick --out runs/run1
uv run ascends shap --run runs/run1
uv run ascends predict --run runs/run1 --csv examples/sample.csv --out runs/run1/predict_01
```
