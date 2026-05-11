"""
======================================
Gap-Filling Method Comparison
======================================

Compare three gap-filling methods: MDS, Random Forest, and XGBoost.

Demonstrates gap-filling workflow comparison with performance metrics and
cumulative carbon flux visualization. Uses one month of data for fast execution.

Methods compared:
- **MDS**: Meteorological similarity, no training required
- **Random Forest**: Machine learning bagging approach
- **XGBoost**: Machine learning gradient boosting approach
"""

# %%
# Gap-filling method comparison
# ==============================
#
# This example demonstrates a complete gap-filling comparison workflow:
# 1. Load one month of example ecosystem flux data
# 2. Gap-fill using MDS method (meteorological similarity)
# 3. Gap-fill using Random Forest (machine learning - bagging)
# 4. Gap-fill using XGBoost (machine learning - boosting)
# 5. Compare results with performance metrics
# 6. Visualize cumulative carbon flux from all three methods

import time
import pandas as pd
import matplotlib.pyplot as plt

import diive as dv
from diive.core.ml.feature_engineer import FeatureEngineer

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df_orig = dv.load_exampledata_parquet()
# Use May 2022 for fast example execution
df = df_orig.loc[(df_orig.index.year == 2022) & (df_orig.index.month == 5)].copy()

# Target and features
TARGET_COL = 'NEE_CUT_REF_orig'
MDS_FEATURES = ['Tair_f', 'Rg_f', 'VPD_f']
RF_FEATURES = ['Tair_f', 'Rg_f', 'VPD_f']

print("\n" + "=" * 80)
print("GAP-FILLING METHOD COMPARISON".center(80))
print("=" * 80)
print(f"Data period: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total records: {len(df)}")
print(f"Missing values (gaps): {df[TARGET_COL].isnull().sum()}")

# %%
# Method 1: MDS (Marginal Distribution Sampling)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MDS fills gaps using average flux from similar meteorological conditions.
# Advantages: No training required, fast. Disadvantages: May miss non-linear patterns.

print("\n" + "-" * 80)
print("METHOD 1: MARGINAL DISTRIBUTION SAMPLING (MDS)".ljust(80))
print("-" * 80)

start_time = time.perf_counter()

mds = dv.FluxMDS(
    df=df,
    flux=TARGET_COL,
    ta='Tair_f',
    swin='Rg_f',
    vpd='VPD_f',
    swin_tol=[20, 50],
    ta_tol=2.5,
    vpd_tol=0.5,
    avg_min_n_vals=5,
    verbose=0
)
mds.run()
mds_gapfilled = mds.get_gapfilled_target()
mds_time = time.perf_counter() - start_time

print(f"Execution time: {mds_time:.2f}s")
mds.report()

# %%
# Method 2: Random Forest
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Machine learning bagging approach. Uses identical feature engineering
# as XGBoost for fair comparison.

print("\n" + "-" * 80)
print("METHOD 2: RANDOM FOREST WITH FEATURE ENGINEERING".ljust(80))
print("-" * 80)

start_time = time.perf_counter()

# Prepare subset with target and RF features
df_rf = df[[TARGET_COL] + RF_FEATURES].copy()

# Create engineered features (production settings for CO2 flux)
engineer = FeatureEngineer(
    target_col=TARGET_COL,
    features_lag=[-2, -1],
    features_lag_stepsize=1,
    features_lag_exclude_cols=None,
    features_rolling=[2, 4, 12, 24, 48],
    features_rolling_exclude_cols=None,
    features_rolling_stats=['mean', 'median', 'min', 'max'],
    features_diff=[1, 2],
    features_diff_exclude_cols=None,
    features_ema=[6, 12, 24, 48],
    features_ema_exclude_cols=None,
    features_poly_degree=2,
    features_poly_exclude_cols=None,
    features_stl=False,
    features_stl_method='stl',
    features_stl_seasonal_period=None,
    features_stl_exclude_cols=None,
    features_stl_components=None,
    vectorize_timestamps=True,
    add_continuous_record_number=False,
    sanitize_timestamp=False
)
df_engineered = engineer.fit_transform(df_rf)

# Create and train RF model
rfts = dv.RandomForestTS(
    input_df=df_engineered,
    target_col=TARGET_COL,
    verbose=0,
    n_estimators=50,
    random_state=42,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1
)

# Train model
rfts.trainmodel(showplot_scores=False, showplot_importance=False)

# Gap-fill
rfts.fillgaps(showplot_scores=False, showplot_importance=False)
rf_gapfilled = rfts.get_gapfilled_target()
rf_time = time.perf_counter() - start_time

print(f"Execution time: {rf_time:.2f}s")
rfts.report_gapfilling()

# %%
# Method 3: XGBoost
# ^^^^^^^^^^^^^^^^^
#
# Machine learning gradient boosting approach. Uses the same engineered features
# as Random Forest for fair comparison.

print("\n" + "-" * 80)
print("METHOD 3: XGBOOST WITH FEATURE ENGINEERING".ljust(80))
print("-" * 80)

start_time = time.perf_counter()

# Reuse engineered features from Random Forest (same features for fair comparison)
xgbts = dv.XGBoostTS(
    input_df=df_engineered,
    target_col=TARGET_COL,
    verbose=0,
    n_estimators=50,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=10,
    n_jobs=-1
)

# Train model
xgbts.trainmodel(showplot_scores=False, showplot_importance=False)

# Gap-fill
xgbts.fillgaps(showplot_scores=False, showplot_importance=False)
xgb_gapfilled = xgbts.get_gapfilled_target()
xgb_time = time.perf_counter() - start_time

print(f"Execution time: {xgb_time:.2f}s")
xgbts.report_gapfilling()

# %%
# Performance comparison
# ^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON".center(80))
print("=" * 80)

comparison_data = {
    'Metric': ['Execution Time (s)', 'Training Required', 'Features Used', 'Approach'],
    'MDS': [f'{mds_time:.2f}', 'No', 'Meteorological', 'Similarity-based'],
    'Random Forest': [f'{rf_time:.2f}', 'Yes', 'Engineered (45+)', 'Bagging'],
    'XGBoost': [f'{xgb_time:.2f}', 'Yes', 'Engineered (45+)', 'Boosting']
}
print("\n" + pd.DataFrame(comparison_data).to_string(index=False))

# %%
# Cumulative carbon flux comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Convert fluxes to carbon units and plot cumulative sums for all three methods.

print("\n" + "-" * 80)
print("CUMULATIVE CARBON FLUX VISUALIZATION".ljust(80))
print("-" * 80)

# Prepare cumulative data
observed = df[TARGET_COL]
df_cumulative = pd.DataFrame({
    'Observed': observed,
    'MDS': mds_gapfilled,
    'Random Forest': rf_gapfilled,
    'XGBoost': xgb_gapfilled
})

# Convert from umol CO2 m-2 s-1 to g C m-2 30min-1
df_cumulative = df_cumulative.multiply(0.02161926)
series_units = r'($\mathrm{gC\ m^{-2}}$)'

# Create cumulative plot
dv.plot_cumulative(
    df=df_cumulative,
    units=series_units,
    start_year=2022,
    end_year=2022
).plot()

# %%
# Time series heatmap comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "-" * 80)
print("TIME SERIES HEATMAP COMPARISON".ljust(80))
print("-" * 80)

fig, axes = plt.subplots(1, 4, figsize=(28, 5),
                         gridspec_kw={'wspace': 0.12},
                         constrained_layout=True)

# Observed
dv.plot_heatmap_datetime(series=observed).plot(ax=axes[0])
axes[0].set_title('Observed\n(with gaps)', fontsize=11, fontweight='bold')

# MDS gap-filled
dv.plot_heatmap_datetime(series=mds_gapfilled).plot(ax=axes[1])
axes[1].set_title('MDS\nGap-Filled', fontsize=11, fontweight='bold')

# RF gap-filled
dv.plot_heatmap_datetime(series=rf_gapfilled).plot(ax=axes[2])
axes[2].set_title('Random Forest\nGap-Filled', fontsize=11, fontweight='bold')

# XGB gap-filled
dv.plot_heatmap_datetime(series=xgb_gapfilled).plot(ax=axes[3])
axes[3].set_title('XGBoost\nGap-Filled', fontsize=11, fontweight='bold')

fig.suptitle('Gap-Filling Method Comparison', fontsize=13, fontweight='bold', y=1.00)
plt.show()

print("✓ Gap-filling method comparison complete.")
