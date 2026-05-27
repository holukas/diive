"""
=====================================
SW_IN Gap-Filling (Physics + XGBoost)
=====================================

Gap-fill shortwave incoming radiation using physics-aware partitioning.

Nighttime gaps are set to zero (no solar radiation after sunset).
Daytime gaps are filled with XGBoost using potential radiation and
engineered temporal features.  The two parts are assembled into one
complete, physically consistent time series.
"""

# %%
# Overview
# ^^^^^^^^
#
# Shortwave incoming radiation (SW_IN) has a hard physical constraint:
# it is exactly zero at night.  A generic gap-filler ignores this and
# can produce small non-zero (or even negative) nighttime predictions.
#
# SWINGapFillerXGBoost avoids this by:
#
# 1. Calculating potential radiation (SW_IN_POT) from lat/lon
# 2. Using SW_IN_POT as the daytime/nighttime divider (threshold 20 W/m2)
# 3. Filling nighttime gaps with zero (physics)
# 4. Filling daytime gaps with XGBoost trained on daytime data only
# 5. Assembling the two parts into one gap-free series
#
# No additional driver variables are needed by default.  SW_IN_POT alone
# (plus lag/rolling features of SW_IN itself and timestamp features)
# is sufficient.  Additional drivers such as TA or VPD can be supplied
# via context_df to improve daytime prediction quality.
#
# An optional nighttime offset correction can be applied first to remove
# the systematic sensor bias that causes small negative values at night.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

# %%
# Site configuration and data loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SITE_LAT = 47.286417   # CH-DAV Davos, Switzerland
SITE_LON = 7.733750
SITE_UTC_OFFSET = 1
TARGET_COL = 'Rg_f'   # Shortwave incoming radiation (W/m2)

df_orig = dv.load_exampledata_parquet()
df = df_orig.copy()
keep = (df.index.year >= 2020) & (df.index.year <= 2020)
df = df[keep].copy()

# Introduce artificial gaps (randomly remove 15% of observed values)
rng = np.random.default_rng(seed=42)
observed_idx = df[TARGET_COL].dropna().index
gap_idx = rng.choice(observed_idx, size=int(0.15 * len(observed_idx)), replace=False)
df.loc[gap_idx, TARGET_COL] = np.nan

print(f"Data loaded: {len(df)} records")
print(f"Missing values in {TARGET_COL}: {df[TARGET_COL].isnull().sum()} "
      f"({100 * df[TARGET_COL].isnull().mean():.1f}%)")

# %%
# Basic usage: SW_IN_POT + timestamps only (no extra drivers needed)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This is the default configuration.  Only potential radiation (SW_IN_POT)
# and timestamp features are used as predictors, together with lag and
# rolling features derived from SW_IN itself.

gf = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    nighttime_threshold=20,    # W/m2: below this = nighttime
    correct_nighttime_offset=False,
    reduce_features=False,
    verbose=1,
    # XGBoost hyperparameters (optional)
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
gf.run()

# %%
# Inspect results
# ^^^^^^^^^^^^^^^^

r = gf.results

gapfilled = r.gapfilled
flag = r.flag

print(f"\nGap-filling complete")
print(f"  Observed (flag=0):               {(flag == 0).sum()}")
print(f"  Gap-filled by XGBoost (flag=1):  {(flag == 1).sum()}")
print(f"  Nighttime set to zero (flag=2):  {(flag == 2).sum()}")

if r.scores_traintest:
    print(f"\nDaytime model performance (train/test split):")
    print(f"  R2:   {r.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r.scores_traintest.get('rmse', float('nan')):.2f} W/m2")
    print(f"  MAE:  {r.scores_traintest.get('mae', float('nan')):.2f} W/m2")

# %%
# SHAP feature importances
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SW_IN_POT should rank first, followed by lag and rolling features of SW_IN.
# No meteorological drivers were needed to achieve strong performance.

if r.feature_importances is not None:
    fi = r.feature_importances.copy()
    print(f"\nTop 10 features by SHAP importance (daytime model):")
    print(fi.head(10).to_string())

# %%
# Optional: include additional driver variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Providing TA and VPD as context drivers lets the XGBoost model learn
# how cloud cover correlates with temperature and humidity conditions.
# This can improve prediction quality when those variables are available.

gf_with_context = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    context_df=df[['Tair_f', 'VPD_f']],
    verbose=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
gf_with_context.run()
r_ctx = gf_with_context.results

if r_ctx.scores_traintest:
    print(f"\nWith TA+VPD context:")
    print(f"  R2:   {r_ctx.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r_ctx.scores_traintest.get('rmse', float('nan')):.2f} W/m2")

# %%
# Optional: nighttime offset correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Some radiation sensors record small non-zero (often negative) values at night
# due to thermal emission or electronic offsets.  Set correct_nighttime_offset=True
# to first remove this bias via remove_radiation_zero_offset(), then gap-fill.
# The corrected input is stored in results.gapfilling_df as '{col}_offset_corrected'.

gf_corrected = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    correct_nighttime_offset=True,
    verbose=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
gf_corrected.run()
r_corr = gf_corrected.results

print(f"\nWith nighttime offset correction:")
print(f"  Columns in gapfilling_df: {list(r_corr.gapfilling_df.columns)}")
if r_corr.scores_traintest:
    print(f"  R2:   {r_corr.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r_corr.scores_traintest.get('rmse', float('nan')):.2f} W/m2")

# %%
# Visualize: observed vs gap-filled heatmaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                         gridspec_kw={'wspace': 0.15},
                         constrained_layout=True)

dv.plotting.HeatmapDateTime(series=df[TARGET_COL]).plot(
    ax=axes[0],
    zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[0].set_title('Observed SW_IN\n(with gaps)', fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gapfilled).plot(
    ax=axes[1],
    zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[1].set_title('Gap-Filled SW_IN\n(SW_IN_POT + XGBoost)', fontsize=11, fontweight='bold')

fig.suptitle('SW_IN Gap-Filling: Physics + XGBoost', fontsize=13, fontweight='bold')
plt.show()

print("SW_IN gap-filling example complete.")
