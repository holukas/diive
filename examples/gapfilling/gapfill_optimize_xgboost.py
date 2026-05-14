"""
=====================================
XGBoost Hyperparameter Optimization
=====================================

Optimize XGBoost gap-filling hyperparameters using GridSearchCV with
comprehensive analysis of parameter importance and performance.

Demonstrates hyperparameter tuning for XGBoost gap-filling with
time series cross-validation. Tests multiple parameter combinations to find
optimal settings, showing: data characteristics, best parameters, model scores,
cross-validation results, and parameter sensitivity plots.

Uses 2020 data only for faster optimization testing.
"""

# %%
# Hyperparameter optimization workflow
# =====================================
#
# Optimize XGBoost hyperparameters using GridSearchCV with
# time series cross-validation. Analyze results to understand which
# parameters have the biggest impact on model performance.

import xgboost as xgb

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

TARGET_COL = 'NEE_CUT_REF_orig'
subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

# Example data
df = dv.load_exampledata_parquet()
subset = df[subsetcols].copy()
_subset = df.index.year == 2020
subset = subset[_subset].copy()

print(f"Data loaded: {len(subset)} records from {subset.index.min().date()} to {subset.index.max().date()}")
print(f"Missing values in target: {subset[TARGET_COL].isnull().sum()}")

# Data statistics and characteristics
print(f"\nTarget variable statistics:")
stats = dv.sstats(subset[TARGET_COL])
print(stats)

# Visualize the target time series
print(f"\nGenerating time series plot...")
ts_plot = dv.TimeSeries(series=subset[TARGET_COL])
ts_plot.plot()

# %%
# Define hyperparameter space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify the parameter grid for testing. Using realistic production-range
# parameters. Larger grid = more combinations but more computational time.
#
# Note: Early stopping in XGBoost requires validation sets to be provided at
# fit-time via eval_set parameter, which is not compatible with standard GridSearchCV.
# This example uses fixed n_estimators. For production, consider training a single
# model with early stopping or using custom GridSearchCV wrappers.

xgb_params = {
    'n_estimators': [10],  # Maximum boosting rounds
    'max_depth': [2, 4, 6],  # Maximum tree depth (deeper = more complex)
    'learning_rate': [0.01, 0.1, 0.5],  # Shrinkage (lower = slower learning, more conservative)
    'subsample': [0.5, 0.9, 1.0],  # Fraction of samples used per boosting round (prevents overfitting)
    'colsample_bytree': [0.5, 0.9, 1.0],  # Fraction of features used per boosting round (prevents overfitting)
    'n_jobs': [-1]  # Use all available CPU cores for parallel processing
}

print(f"\nHyperparameter space to explore:")
total_combinations = 1
for param, values in xgb_params.items():
    print(f"  {param}: {values}")
    total_combinations *= len(values)
print(f"  Total combinations: {total_combinations}")
print(f"  (With 10 CV folds: {total_combinations * 10} model fits)")

# %%
# Run optimization
# ^^^^^^^^^^^^^^^^
#
# Execute GridSearchCV with time series cross-validation to find
# the best parameter combination. Each parameter combination is tested
# across 10 time series cross-validation folds.

print(f"\n" + "=" * 80)
print("Starting hyperparameter optimization...")
print("=" * 80)

opt = dv.OptimizeParamsTS(
    df=subset,
    target_col=TARGET_COL,
    regressor_class=xgb.XGBRegressor,
    **xgb_params
)

opt.optimize()

# %%
# Best parameters and model performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display the optimal hyperparameters found and the resulting model scores.
# The best parameters are those with the highest mean cross-validated score
# across all tested combinations.

print(f"\n" + "=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)

print(f"\nBest hyperparameters:")
for param, value in opt.best_params.items():
    print(f"  {param}: {value}")

print(f"\nBest model scores:")
for metric, value in opt.scores.items():
    print(f"  {metric}: {value:.4f}")

print(f"\nMean cross-validated score: {opt.best_score:.4f}")
print(f"Cross-validation folds: {opt.cv_n_splits}")

# %%
# Comprehensive results report
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display detailed optimization report with top parameter combinations
# and their performance metrics.

opt.report_optimization(top_n=3)

# %%
# Cross-validation results table
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show all parameter combinations ranked by test score,
# with detailed performance metrics for each.

print(f"\nAll parameter combinations ranked by test score:")
cv_summary = opt.cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].copy()
cv_summary = cv_summary.sort_values('rank_test_score').head(10)
print(cv_summary.to_string(index=False))

# %%
# Parameter sensitivity analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize how each hyperparameter affects model performance.
# Shows which parameters have the biggest impact on cross-validation score.

print(f"\nGenerating optimization visualization...")

# %%
# Optimization visualization using built-in class methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Call the built-in plotting methods on the OptimizeParamsTS instance to generate
# comprehensive visualizations showing convergence, parameter importance, parameter slices,
# and parallel coordinates.

opt.plot_optimization_analysis()

# %%
# Summary and recommendations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The optimization process explored all tested parameter combinations
# using time series cross-validation. The best model configuration
# and production-ready code are shown above.

print(f"\n" + "=" * 80)
print("Hyperparameter optimization complete!")
print("=" * 80)

print(f"\n[OK] Best parameters identified and ready for model training")
print(f"[OK] Model scores show R² = {opt.scores.get('r2', 'N/A'):.4f}")
print(f"[OK] Use opt.best_params to initialize XGBRegressor for production")
