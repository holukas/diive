"""
=====================================
Random Forest Hyperparameter Optimization
=====================================

Optimize Random Forest gap-filling hyperparameters using GridSearchCV with
comprehensive analysis of parameter importance and performance.

Demonstrates hyperparameter tuning for Random Forest gap-filling with
time series cross-validation. Tests multiple parameter combinations to find
optimal settings, showing: data characteristics, best parameters, model scores,
cross-validation results, and parameter sensitivity plots.

Uses 2020 data only for faster optimization testing.
"""

# %%
# Hyperparameter optimization workflow
# =====================================
#
# Optimize Random Forest hyperparameters using GridSearchCV with
# time series cross-validation. Analyze results to understand which
# parameters have the biggest impact on model performance.

from sklearn.ensemble import RandomForestRegressor

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
ts_plot = dv.plotting.TimeSeries(series=subset[TARGET_COL])
ts_plot.plot()

# %%
# Define hyperparameter space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify the parameter grid for testing. Using realistic production-range
# parameters. Larger grid = more combinations but more computational time.

rf_params = {
    'n_estimators': [1, 3, 5, 7],
    'max_features': [1],
    'criterion': ['squared_error'],
    'max_depth': [2, 4, None],
    'min_samples_split': [2, 20],
    'min_samples_leaf': [1, 10],
}

print(f"\nHyperparameter space to explore:")
total_combinations = 1
for param, values in rf_params.items():
    print(f"  {param}: {values}")
    total_combinations *= len(values)
print(f"  Total combinations: {total_combinations}")
print(f"  (With 10 CV folds: {total_combinations * 10} model fits)")

# %%
# Run optimization
# ^^^^^^^^^^^^^^^^
#
# Execute GridSearchCV with time series cross-validation to find
# the best parameter combination.

print(f"\n" + "=" * 80)
print("Starting hyperparameter optimization...")
print("=" * 80)

opt = dv.gapfilling.OptimizeParamsTS(
    df=subset,
    target_col=TARGET_COL,
    regressor_class=RandomForestRegressor,
    **rf_params
)

opt.optimize()

# %%
# Best parameters and model performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display the optimal hyperparameters found and the resulting model scores.

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
print(f"[OK] Use opt.best_params to initialize RandomForestRegressor for production")
