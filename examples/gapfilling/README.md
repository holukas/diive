# Gap-Filling Methods Examples

Examples demonstrating various gap-filling approaches for time series data, from simple to advanced machine learning.

12 examples across 5 gap-filling methods with optimization and comparison workflows.

## Method Overview

Linear interpolation is fast but works only for small gaps. Random Forest and XGBoost require training data but handle larger gaps and complex patterns. MDS (Meteorological Data Similarity) needs no training — it matches similar conditions across your dataset. SW_IN Physics+XGBoost uses solar geometry to constrain nighttime values to zero and fills daytime gaps with gradient boosting.

| Method | Training | Best For |
|--------|----------|----------|
| Linear Interpolation | No | Small gaps (<1 day) |
| Random Forest | Yes | General purpose, handles nonlinear patterns |
| XGBoost | Yes | High accuracy, best for large datasets |
| MDS | No | When you lack training data |
| SW_IN Physics+XGBoost | Yes | Shortwave radiation with physical nighttime constraint |

## Examples by Method

### Linear Interpolation

Simple, no training required. Works for small gaps.

- **gapfill_interpolate_conservative.py** — Strict: max gap length 1
- **gapfill_interpolate_generous.py** — Permissive: max gap length 5

### Random Forest

Training-based, interpretable, robust to outliers. Four versions: basic, long-term year-pooling, quick prototype, and hyperparameter-tuned.

- **gapfill_randomforest.py** — Basic Random Forest with 8-stage feature engineering
- **gapfill_randomforest_longterm.py** — Long-term gap-filling using year-pooling strategy (optimal for multi-year datasets)
- **gapfill_quickfill.py** — Quick prototype (faster for exploration)
- **gapfill_optimize_randomforest.py** — Hyperparameter tuning via grid search

### XGBoost

Gradient boosting. Often more accurate than Random Forest but requires more tuning.

- **gapfill_xgboost.py** — Basic XGBoost with default hyperparameters
- **gapfill_optimize_xgboost.py** — Hyperparameter tuning via grid search

### MDS (Meteorological Data Similarity)

No training. Fills gaps by finding similar conditions elsewhere in your data.

- **gapfill_mds.py** — Original MDS implementation
- **gapfill_mds_comparison.py** — MDS reproducibility / determinism check (two runs are bit-identical)

### SW_IN Physics + XGBoost

Physics-constrained gap-filling for shortwave incoming radiation. Nighttime values are always set to zero; daytime gaps are filled with XGBoost trained on potential radiation and timestamp features. Only lat/lon/UTC offset required — no meteorological driver variables needed by default.

- **gapfill_swin.py** — SW_IN gap-filling: physics sets nighttime to zero, XGBoost fills daytime gaps. Runs three configurations, each differing from the first by one setting — the defaults (the climatology ceiling of a timestamp-only model), plus `reduce_features=True`, plus a second radiation sensor (PPFD) passed via `context_df`, which is what breaks the ceiling

### Comparison & Benchmarking

- **gapfill_comparison.py** — Run MDS, Random Forest, and XGBoost on the same data, compare R², MAE, RMSE, and runtime

## When to Use Each Method

**Linear interpolation:** Your gaps are small (a few hours or less) and you don't need high accuracy.

```python
import diive as dv

filled = dv.gapfilling.linear_interpolation(series=df['NEE'], limit=1)
```

**Random Forest or XGBoost:** You have training data and want good accuracy without excessive tuning. Start with Random Forest for interpretability; switch to XGBoost if you need better accuracy on a specific dataset.

```python
import diive as dv

# Features are whatever columns `df` holds besides the target, plus everything
# the engineer derives from them. There is no `features` argument on the model.
engineer = dv.gapfilling.FeatureEngineer(
    target_col='NEE',
    features_lag=[-2, -1],
    features_rolling=[12, 24],
    features_ema=[6, 12],
    vectorize_timestamps=True
)
df_engineered = engineer.fit_transform(df)

model = dv.gapfilling.RandomForestTS(
    input_df=df_engineered,
    target_col='NEE',
    n_estimators=500
)
model.trainmodel()   # train on complete observations
model.fillgaps()     # predict the missing values
gapfilled = model.get_gapfilled_target()
```

Keyword arguments other than `input_df`, `target_col`, `verbose`, `test_size` and
`below_zero` are passed on to the underlying regressor (e.g. `n_estimators`,
`random_state`, `n_jobs`).

**MDS:** You have no training data, or you want to avoid potential overfitting from learned models.

```python
import diive as dv

# VPD is expected in kPa (`vpd_in_kpa=True` by default), SW_IN in W/m2, TA in degC.
mds = dv.gapfilling.FluxMDS(
    df=df,
    flux='NEE',
    swin='SW_IN',
    ta='TA',
    vpd='VPD',
    swin_tol=[20, 50],
    ta_tol=2.5,
    vpd_tol=0.5
)
mds.run()
filled = mds.get_gapfilled_target()
```

## Running Examples

```bash
uv run python examples/gapfilling/gapfill_comparison.py
```

For individual methods:

```bash
# Linear interpolation
uv run python examples/gapfilling/gapfill_interpolate_conservative.py
uv run python examples/gapfilling/gapfill_interpolate_generous.py

# Random Forest
uv run python examples/gapfilling/gapfill_randomforest.py
uv run python examples/gapfilling/gapfill_randomforest_longterm.py
uv run python examples/gapfilling/gapfill_quickfill.py
uv run python examples/gapfilling/gapfill_optimize_randomforest.py

# XGBoost
uv run python examples/gapfilling/gapfill_xgboost.py
uv run python examples/gapfilling/gapfill_optimize_xgboost.py

# MDS
uv run python examples/gapfilling/gapfill_mds.py
uv run python examples/gapfilling/gapfill_mds_comparison.py

# SW_IN physics + XGBoost
uv run python examples/gapfilling/gapfill_swin.py

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

See `dv.gapfilling` for API documentation:
- `RandomForestTS` — Random Forest time series with 8-stage feature engineering
- `XGBoostTS` — XGBoost gradient boosting with tunable hyperparameters
- `linear_interpolation()` — Simple linear interpolation
- `FluxMDS` — Meteorological Data Similarity
