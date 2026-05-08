# Gap-Filling Methods Examples

Examples demonstrating various gap-filling approaches for time series data, including machine learning and statistical methods.

## Contents

- **interpolate.py** — Linear interpolation for small gaps
- **randomforest_ts.py** — Random Forest gap-filling with feature engineering
- **xgboost_ts.py** — XGBoost gradient boosting gap-filling
- **mds.py** — Meteorological Data Similarity (MDS) method
- **mds_comparison.py** — Comparison of MDS variants
- **comparison.py** — Benchmarking and comparing multiple gap-filling methods

## Related Documentation

See `diive.pkgs.gapfilling` for:
- `RandomForestTS` — Random Forest time series gap-filling
- `XGBoostTS` — XGBoost time series gap-filling
- `linear_interpolation()` — Simple linear interpolation
- `_FluxMDS` — MDS method implementation

## Running Examples

```bash
uv run python examples/pkgs/gapfilling/randomforest_ts.py
uv run python examples/pkgs/gapfilling/comparison.py
```

Or run all gap-filling examples:

```bash
uv run python examples/run_all_examples.py
```
