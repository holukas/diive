# Gap-Filling Examples

Machine learning and statistical methods for gap-filling in time series data.

## Overview

Comprehensive gap-filling workflows with multiple algorithms:

- **Random Forest Gap-Filling** — ML-based gap-filling with 8-stage feature engineering (R² 0.60-0.80)
- **XGBoost Gap-Filling** — Gradient boosting approach with hyperparameter optimization (R² 0.65-0.85)
- **Marginal Distribution Sampling (MDS)** — Meteorological similarity matching, no training required
- **Linear Interpolation** — Simple conservative gap-filling for small gaps
- **Method Comparison** — Side-by-side evaluation of ML vs. MDS approaches
- **Feature Engineering** — Harmonized 8-stage pipeline: lags, rolling stats, differencing, EMA, polynomial terms, STL, timestamps, record number

## Use Cases

- Filling flux measurement gaps in eddy covariance data
- Preparing gap-free datasets for ecosystem analyses
- Comparing gap-filling method performance
- Long-term flux dataset completion
