# Gap-Filling Methods Examples

Examples demonstrating various gap-filling approaches for time series data, from simple to advanced machine learning.

10 examples across 4 gap-filling methods with optimization and comparison workflows.

## Method Overview

Linear interpolation is fast but works only for small gaps. Random Forest and XGBoost require training data but handle larger gaps and complex patterns. MDS (Meteorological Data Similarity) needs no training — it matches similar conditions across your dataset.

| Method | Training | Best For |
|--------|----------|----------|
| Linear Interpolation | No | Small gaps (<1 day) |
| Random Forest | Yes | General purpose, handles nonlinear patterns |
| XGBoost | Yes | High accuracy, best for large datasets |
| MDS | No | When you lack training data |

## Examples by Method

### Linear Interpolation

Simple, no training required. Works for small gaps.

- **gapfill_interpolate_conservative.py** — Strict: max gap length 1
- **gapfill_interpolate_generous.py** — Permissive: max gap length 5

### Random Forest

Training-based, interpretable, robust to outliers. Three versions: basic, quick prototype, and hyperparameter-tuned.

- **gapfill_randomforest.py** — Basic Random Forest with 8-stage feature engineering
- **gapfill_quickfill.py** — Quick prototype (faster for exploration)
- **gapfill_optimize_randomforest.py** — Hyperparameter tuning via grid search

### XGBoost

Gradient boosting. Often more accurate than Random Forest but requires more tuning.

- **gapfill_xgboost.py** — Basic XGBoost with default hyperparameters
- **gapfill_optimize_xgboost.py** — Hyperparameter tuning via grid search

### MDS (Meteorological Data Similarity)

No training. Fills gaps by finding similar conditions elsewhere in your data.

- **gapfill_mds.py** — Original MDS implementation
- **gapfill_mds_comparison.py** — Comparison of original vs. optimized variants

### Comparison & Benchmarking

- **gapfill_comparison.py** — Run all 4 methods on the same data, compare R², MAE, RMSE, and runtime

## When to Use Each Method

**Linear interpolation:** Your gaps are small (a few hours or less) and you don't need high accuracy.

```python
from diive.pkgs.gapfilling import linear_interpolation

filled = linear_interpolation(series=df['NEE'], limit=1)
```

**Random Forest or XGBoost:** You have training data and want good accuracy without excessive tuning. Start with Random Forest for interpretability; switch to XGBoost if you need better accuracy on a specific dataset.

```python
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.gapfilling import RandomForestTS

engineer = FeatureEngineer(
    target_col='NEE',
    features_lag=[-2, -1],
    features_rolling=[12, 24],
    features_ema=[6, 12],
    vectorize_timestamps=True
)
df_engineered = engineer.fit_transform(df)

model = RandomForestTS(
    input_df=df_engineered,
    target_col='NEE',
    n_estimators=500,
    features=['TA', 'VPD', 'SW_IN']
)
model.trainmodel()
gapfilled = model.get_gapfilled_target()
```

**MDS:** You have no training data, or you want to avoid potential overfitting from learned models.

```python
from diive.pkgs.gapfilling import FluxMDS

mds = FluxMDS(
    df=df,
    flux_col='NEE',
    swin_col='SW_IN',
    ta_col='TA',
    vpd_col='VPD',
    swin_tol=[20, 50],
    ta_tol=2.5,
    vpd_tol=0.5
)
filled = mds.get_mds_filled()
```

## Running Examples

```bash
uv run python examples/pkgs/gapfilling/gapfill_comparison.py
```

For individual methods:

```bash
# Linear interpolation
uv run python examples/pkgs/gapfilling/gapfill_interpolate_conservative.py
uv run python examples/pkgs/gapfilling/gapfill_interpolate_generous.py

# Random Forest
uv run python examples/pkgs/gapfilling/gapfill_randomforest.py
uv run python examples/pkgs/gapfilling/gapfill_quickfill.py
uv run python examples/pkgs/gapfilling/gapfill_optimize_randomforest.py

# XGBoost
uv run python examples/pkgs/gapfilling/gapfill_xgboost.py
uv run python examples/pkgs/gapfilling/gapfill_optimize_xgboost.py

# MDS
uv run python examples/pkgs/gapfilling/gapfill_mds.py
uv run python examples/pkgs/gapfilling/gapfill_mds_comparison.py

# All examples
uv run python examples/run_all_examples.py
```

## Feature Engineering

Random Forest and XGBoost both use an identical 8-stage feature engineering pipeline:

1. **Lag features** — Past/future values (e.g., [-2, -1])
2. **Rolling statistics** — Moving mean, median, min, max, std (e.g., windows=[12, 24, 48])
3. **Differencing** — Rate of change (1st and 2nd order)
4. **Exponential Moving Average** — EMA decay (e.g., [6, 12, 24])
5. **Polynomial features** — Squared/cubic terms (e.g., degree=2)
6. **STL decomposition** — Trend, seasonal, residual (optional)
7. **Timestamps** — Year, month, hour, season (creates ~19 features for diurnal/seasonal patterns)
8. **Record number** — Continuous ordering to detect long-term drift

See `diive.core.ml.feature_engineer.FeatureEngineer` for details.

## Related Documentation

See `diive.pkgs.gapfilling` for API documentation:
- `RandomForestTS` — Random Forest time series with 8-stage feature engineering
- `XGBoostTS` — XGBoost gradient boosting with tunable hyperparameters
- `linear_interpolation()` — Simple linear interpolation
- `FluxMDS` — Meteorological Data Similarity
