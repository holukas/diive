"""
=================================
XGBoost Hyperparameter Optimization
=================================

Optimize XGBoost gap-filling hyperparameters using GridSearchCV.

Demonstrates hyperparameter tuning for XGBoost gap-filling with
time series cross-validation. Tests multiple parameter combinations to find
optimal settings for maximum gap-filling accuracy.

Uses 2020 data only for faster optimization testing.
"""

# %%
# Hyperparameter optimization workflow
# =====================================
#
# Optimize XGBoost hyperparameters using GridSearchCV with
# time series cross-validation to find the best parameter combinations.

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
_subset = df.index.year >= 2020
subset = subset[_subset].copy()

print(f"Data loaded: {len(subset)} records from {subset.index.min().date()} to {subset.index.max().date()}")
print(f"Missing values in target: {subset[TARGET_COL].isnull().sum()}")

# %%
# Define hyperparameter space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify the parameter grid for testing. Using minimal combinations for speed:
# ~20 combinations × 10 CV folds = 200 model fits.

xgb_params = {
    'n_estimators': [30, 50],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
}

print(f"\nHyperparameter space to explore:")
for param, values in xgb_params.items():
    print(f"  {param}: {values}")

# %%
# Run optimization
# ^^^^^^^^^^^^^^^^
#
# Execute GridSearchCV with time series cross-validation.

opt = dv.OptimizeParamsTS(
    df=subset,
    target_col=TARGET_COL,
    regressor_class=xgb.XGBRegressor,
    **xgb_params
)

opt.optimize()

# %%
# Print results and recommendations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display comprehensive optimization report with top 3 parameter sets.

opt.report_optimization(top_n=3)

print("✓ Hyperparameter optimization complete.")
