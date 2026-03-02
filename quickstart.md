# Quickstart

## Easiest path

```bash
./install.sh
./run_gui.sh
```

Then open:

`http://127.0.0.1:7777`

## CLI examples

```bash
uv run ascends correlation --csv examples/BostonHousing.csv --target medv --task r --view wide
uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf
uv run ascends parity-plot runs/boston_rf --scope combined --out runs/boston_rf
uv run ascends predict runs/boston_rf --csv examples/BostonHousing_test.csv --out runs/boston_rf/predict
```
