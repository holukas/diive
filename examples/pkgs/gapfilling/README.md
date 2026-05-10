# Gap-Filling Methods Examples

Examples demonstrating various gap-filling approaches for time series data, including machine learning and statistical methods.

## Contents

### Linear Interpolation
- **gapfill_interpolate_generous.py** — Linear interpolation with generous gap limit (limit=5)
- **gapfill_interpolate_conservative.py** — Linear interpolation with conservative limit (limit=1)

### Machine Learning Methods
- **gapfill_randomforest.py** — Random Forest gap-filling with feature engineering
- **gapfill_quickfill.py** — QuickFill: Rapid Random Forest prototyping
- **gapfill_optimize_randomforest.py** — Hyperparameter optimization for Random Forest
- **gapfill_xgboost.py** — XGBoost gradient boosting gap-filling
- **gapfill_optimize_xgboost.py** — Hyperparameter optimization for XGBoost

### Meteorological Data Similarity
- **gapfill_mds.py** — Marginal Distribution Sampling (MDS) method
- **gapfill_mds_comparison.py** — Original vs optimized MDS performance comparison

### Comparison & Evaluation
- **gapfill_comparison.py** — Benchmark all three methods (MDS, Random Forest, XGBoost) side-by-side

## Related Documentation

See `diive.pkgs.gapfilling` for:
- `RandomForestTS` — Random Forest time series gap-filling
- `XGBoostTS` — XGBoost time series gap-filling
- `linear_interpolation()` — Simple linear interpolation
- `_FluxMDS` — MDS method implementation

## Running Examples

```bash
# Linear interpolation
uv run python examples/pkgs/gapfilling/gapfill_interpolate_generous.py
uv run python examples/pkgs/gapfilling/gapfill_interpolate_conservative.py

# Random Forest (production, quick prototyping, optimization)
uv run python examples/pkgs/gapfilling/gapfill_randomforest.py
uv run python examples/pkgs/gapfilling/gapfill_quickfill.py
uv run python examples/pkgs/gapfilling/gapfill_optimize_randomforest.py

# XGBoost (production, optimization)
uv run python examples/pkgs/gapfilling/gapfill_xgboost.py
uv run python examples/pkgs/gapfilling/gapfill_optimize_xgboost.py

# MDS method and comparison
uv run python examples/pkgs/gapfilling/gapfill_mds.py
uv run python examples/pkgs/gapfilling/gapfill_mds_comparison.py

# Compare all methods
uv run python examples/pkgs/gapfilling/gapfill_comparison.py
```

Or run all gap-filling examples:

```bash
uv run python examples/run_all_examples.py
```
