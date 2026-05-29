"""
===========================
Random Forest Gap-Filling
===========================

Gap-fill time series using Random Forest with feature engineering.

Random Forest is a robust, interpretable machine learning approach for gap-filling
time series data. Effective for non-linear relationships and feature interactions.
Demonstrates the complete workflow: feature engineering, model training, SHAP
importance analysis, and gap-filling predictions.
"""

# %%
# Random Forest gap-filling for CO₂ flux (NEE)
# ==============================================
#
# This example demonstrates the full Random Forest gap-filling workflow:
# 1. Load example ecosystem flux data
# 2. Create engineered features (lag, rolling stats, STL, timestamps)
# 3. Train RandomForestTS model on complete observations
# 4. Evaluate feature importance using SHAP
# 5. Predict missing values (gap-fill)
#
# Features used: Tair_f (temperature), VPD_f (vapor pressure deficit),
# Rg_f (radiation)

import matplotlib.pyplot as plt
import pandas as pd

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

TARGET_COL = 'NEE_CUT_REF_orig'
subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

df_orig = dv.load_exampledata_parquet()
df = df_orig.copy()
keep = (df.index.year >= 2020) & (df.index.year <= 2020)
df = df[keep].copy()
df = df[subsetcols].copy()

print(f"Data loaded: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values in target: {df[TARGET_COL].isnull().sum()}")

# %%
# Engineer features
# ^^^^^^^^^^^^^^^^^
#
# Create an 8-stage feature engineering pipeline to provide temporal context and
# capture non-linear patterns. Each parameter is optional (default None), allowing
# task-specific customization. Choose combinations based on your data characteristics:
#
# - **Periodic/seasonal data**: Use vectorize_timestamps + features_rolling
# - **Trending data**: Use add_continuous_record_number + features_ema
# - **Non-stationary flux data**: Use features_lag + features_diff + features_ema
# - **Non-linear relationships**: Use features_poly_degree
#
# **Parameter Explanations:**
#
# *features_lag* (temporal autocorrelation): Captures past/future values encoding
# temporal persistence. Essential for time series but reduces training samples near
# gaps (NaN inflation) and expands feature count.
#
# *features_rolling* (local envelope/volatility): Captures recent baseline level and
# variability. Mean reveals recent trend; std reveals uncertainty. Computationally
# cheap, never introduces NaN.
#
# *features_rolling_stats* (distribution shape): Beyond mean/std. Percentiles and
# min/max reveal distribution shape and plausible range, robust to outliers.
#
# *features_diff* (rate of change/momentum): Crucial for transient events like
# sunrise/sunset ramps. De-trends data and captures velocity, but loses level
# information and amplifies noise.
#
# *features_ema* (adaptive baseline): For non-stationary time series, EMA tracks
# moving targets better than fixed windows. Smooths noise while respecting trends,
# but requires parameter tuning and less robust to outliers than median-based stats.
#
# *features_poly_degree* (non-linear relationships): Many ecosystem processes are
# non-linear (e.g., radiation via square-law). Enables model to learn curved
# relationships directly, but explodes feature count and prone to overfitting at
# degree-3+.
#
# *vectorize_timestamps* (seasonal/diurnal cycles): Essential for multi-year datasets.
# Adds ~6 features (year, season, month, week, doy, hour) capturing temporal phase.
# Diurnal cycle (hour) often most predictive for ecosystem fluxes.
#
# *add_continuous_record_number* (long-term drift): Allows model to capture linear
# temporal trends (measurement stability, environmental state drift) without explicit
# detrending. Only useful for monotonic trends.

engineer = dv.gapfilling.FeatureEngineer(
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
    features_stl=True,
    features_stl_method='stl',
    features_stl_seasonal_period=None,
    features_stl_exclude_cols=None,
    features_stl_components=None,
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    sanitize_timestamp=False
)
df_engineered = engineer.fit_transform(df)

print(f"\nEngineered features created: {df_engineered.shape[1]} columns")

# %%
# Train Random Forest model
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a RandomForestTS model with the engineered features. The model will:
# 1. Automatically detect features to use
# 2. Reduce features using SHAP importance (keep only important ones)
# 3. Train on complete observations only

rfts = dv.gapfilling.RandomForestTS(
    input_df=df_engineered,
    target_col=TARGET_COL,
    verbose=1,
    below_zero=None,  # How to treat negative predictions: None=keep, 'zero'=clip to 0
    # Use 'zero' or 'nan' for variables that cannot be negative (e.g. VPD, SW_IN, PPFD).
    # NEE can be negative (carbon uptake), so None is correct here.
    n_estimators=3,
    random_state=42,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1
)

# Feature reduction using SHAP importance
rfts.reduce_features(shap_threshold_factor=0.5)
rfts.report_feature_reduction()

# %%
# Train and evaluate model
# ^^^^^^^^^^^^^^^^^^^^^^^^

rfts.trainmodel(showplot_scores=False, showplot_importance=False)
rfts.report_traintest()

# %%
# Gap-fill missing data
# ^^^^^^^^^^^^^^^^^^^^^
#
# Use the trained model to predict missing flux values.

rfts.fillgaps(showplot_scores=False, showplot_importance=False)
rfts.report_gapfilling()

# %%
# Visualize results
# ^^^^^^^^^^^^^^^^^
#
# Compare observed data (with gaps) against Random Forest gap-filled predictions
# using side-by-side heatmaps.

observed = df[TARGET_COL]
gapfilled = rfts.get_gapfilled_target()

fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                         gridspec_kw={'wspace': 0.15},
                         constrained_layout=True)

dv.plotting.HeatmapDateTime(series=observed).plot(ax=axes[0])
axes[0].set_title('Observed\n(with gaps)', fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gapfilled).plot(ax=axes[1])
axes[1].set_title('Random Forest\nGap-Filled', fontsize=11, fontweight='bold')

fig.suptitle('Random Forest Gap-Filling Comparison', fontsize=13, fontweight='bold', y=1.00)
plt.show()

# %%
# Cumulative carbon flux
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Convert fluxes to carbon units and plot cumulative sums to evaluate
# overall agreement between observed and gap-filled data.

df_cumulative = pd.DataFrame({
    'Observed': observed,
    'Gap-filled': gapfilled
})
# Convert from µmol CO₂ m⁻² s⁻¹ to g C m⁻² 30min⁻¹
# Conversion factor: (12 g C / mol) * (1 mol / 1e6 µmol) * (1800 s / 30min)
# = 0.02161926 g C m⁻² 30min⁻¹ per µmol CO₂ m⁻² s⁻¹
df_cumulative = df_cumulative.multiply(0.02161926)
series_units = r'($\mathrm{gC\ m^{-2}}$)'

dv.plotting.Cumulative(
    df=df_cumulative,
    units=series_units,
    start_year=2020,
    end_year=2020
).plot()

print("✓ Random Forest gap-filling complete.")
