"""
========================================================
Random Forest Gap-Filling with Long-Term Year Pooling
========================================================

Gap-fill time series using separate annual random forest models trained with
neighboring years. This "long-term" approach adapts to seasonal patterns and
yearly variability without assuming the data are stationary.

Demonstrates building separate models for each year, where each model is trained
on data from the target year plus 2 neighboring years (e.g., 2020 model uses
2019, 2020, 2021 data).

Best for: Multi-year datasets where seasonal/yearly variation is significant.
"""

# %%
# Long-term random forest gap-filling approach
# ==============================================
#
# The LongTermGapFillingRandomForestTS class enables yearly models that adapt
# to seasonal and inter-annual variation. Each year gets its own trained model
# using a "neighboring years" pool strategy.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv
from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

# Use 3 years of data for speed (demonstrates long-term concept without full 10-year run)
df = df.loc[df.index.year.isin([2020, 2021, 2022])].copy()

# Select target and features
TARGET_COL = 'NEE_CUT_REF_orig'
feature_cols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
df = df[feature_cols].copy()

# Quality control: remove low-quality NEE records
lowquality = df.index.get_level_values(0).to_frame()['QCF_NEE'] > 0 if 'QCF_NEE' in df.columns else False
if isinstance(lowquality, bool) and not lowquality:
    # Fallback: mark some records as missing manually if QCF not available
    pass

print(f"Data: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values: {df[TARGET_COL].isnull().sum()}")

# %%
# Feature engineering for all records
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Before model training, create engineered features (lags, timestamps, trends).
# This preparation applies to all years equally.

engineer = dv.FeatureEngineer(
    target_col=TARGET_COL,
    verbose=2,
    features_lag=[-1, -1],  # 1-step lag on all features except target
    add_continuous_record_number=True,  # Capture long-term drift
    sanitize_timestamp=False  # Example data is already clean; skip 10-step validation
)
df_engineered = engineer.fit_transform(df)

print(f"Engineered features shape: {df_engineered.shape}")

# %%
# Initialize long-term model with yearly pools
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create one random forest model per year. Each model is trained on:
# - The target year's data
# - Plus 2 neighboring years (year-1 and year+1)
#
# This neighboring-years strategy allows each year's model to capture that
# year's seasonal patterns while benefiting from adjacent years' data.

gf = LongTermGapFillingRandomForestTS(
    input_df=df_engineered,
    target_col=TARGET_COL,
    verbose=2,
    below_zero=None,  # How to treat negative predictions: None=keep, 'zero'=clip, 'nan'=set missing
    # Use 'zero' or 'nan' for variables that cannot be negative (e.g. VPD, SW_IN, PPFD).
    # NEE can be negative (carbon uptake), so None is correct here.
    n_estimators=3,  # Reduced from standard 300 for speed (demo only)
    random_state=42,
    n_jobs=-1
)

# Create yearpools: each year gets its target year +/- 1 neighbors
gf.create_yearpools()

# Initialize models for each year using their respective data pools
gf.initialize_yearly_models()

# %%
# Feature reduction across all years (optional)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Identifies features that are important across all years. Fits one model per
# year using SHAP, prunes weak features, then re-initialises the year pools.
# Skipped here for speed (only 3 features + n_estimators=3 leaves nothing to prune).
# Enable in production when working with many engineered features.
#
# gf.reduce_features_across_years()
# print(f"Features retained across all years: {gf.features_reduced_across_years}")

# %%
# Fill gaps with year-specific models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Apply each year's trained model to fill missing values in that year.

gf.fillgaps()

print(f"Gap-filling complete. Gapfilled series: {gf.gapfilled_.name}")

# %%
# Examine per-year performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each year has its own R^2, MAE, and other metrics showing how well
# that year's model performs.

from statistics import mean

mae_list = []
r2_list = []

print("\nPER-YEAR MODEL PERFORMANCE:")
print(f"{'Year':<6} {'R²':<10} {'MAE':<10}")
print("-" * 26)

for year in sorted(gf.scores_.keys()):
    scores = gf.scores_[year]
    mae_list.append(scores['mae'])
    r2_list.append(scores['r2'])
    print(f"{year:<6} {scores['r2']:<10.4f} {scores['mae']:<10.4f}")

print("-" * 26)
print(f"{'MEAN':<6} {mean(r2_list):<10.4f} {mean(mae_list):<10.4f}")

# %%
# Feature importance across years
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The feature importance varies by year, showing how seasonal factors shift.
# For example, radiation (Rg_f) is always top predictor, but its relative
# importance and the rank of other features change across months/years.

feature_importance_df = gf.feature_importance_per_year
print("\nTOP 5 FEATURES BY YEAR:")
print(feature_importance_df.head(5))

# %%
# Visualization: before and after gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axs = plt.subplots(ncols=2, figsize=(18, 9))
dv.plot_heatmap_datetime(series=df[TARGET_COL]).plot(ax=axs[0])
dv.plot_heatmap_datetime(series=gf.gapfilling_df_[gf.gapfilled_.name]).plot(ax=axs[1])

axs[0].set_title("Observed NEE (with gaps)", fontsize=14, fontweight='bold')
axs[1].set_title("Gap-Filled NEE (yearly RF models)", fontsize=14, fontweight='bold')
axs[0].tick_params(labelleft=False)
axs[1].tick_params(labelleft=False)

fig.show()

print("✓ Long-term random forest gap-filling complete.")
