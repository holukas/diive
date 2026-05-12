"""
=====================================
Random Forest Hyperparameter Optimization
=====================================

Optimize Random Forest gap-filling hyperparameters using GridSearchCV.

Demonstrates hyperparameter tuning for Random Forest gap-filling with
time series cross-validation. Tests multiple parameter combinations to find
optimal settings for maximum gap-filling accuracy.

Uses 2020 data only for faster optimization testing.
"""

# %%
# Hyperparameter optimization workflow
# =====================================
#
# Optimize Random Forest hyperparameters using GridSearchCV with
# time series cross-validation to find the best parameter combinations.

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

rf_params = {
    'n_estimators': [3, 6],
    'max_depth': [4, 8],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
}

print(f"\nHyperparameter space to explore:")
for param, values in rf_params.items():
    print(f"  {param}: {values}")

# %%
# Run optimization
# ^^^^^^^^^^^^^^^^
#
# Execute GridSearchCV with time series cross-validation.

opt = dv.OptimizeParamsTS(
    df=subset,
    target_col=TARGET_COL,
    regressor_class=RandomForestRegressor,
    **rf_params
)

opt.optimize()

# %%
# Print results and recommendations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display comprehensive optimization report with top 3 parameter sets.

opt.report_optimization(top_n=3)

print("✓ Hyperparameter optimization complete.")
