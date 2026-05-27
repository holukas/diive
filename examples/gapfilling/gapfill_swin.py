"""
=====================================
SW_IN Gap-Filling (Physics + XGBoost)
=====================================

Gap-fill shortwave incoming radiation using physics-aware partitioning
combined with a nighttime sensor-offset correction.

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
# it is exactly zero at night.  Most pyranometers, however, record small
# non-zero (often slightly negative) values at night due to thermal
# emission of the sensor body or electronic offsets, and a generic
# gap-filler will happily reproduce these artefacts.
#
# SWINGapFillerXGBoost addresses both issues at once:
#
# 1. Calculate potential radiation (SW_IN_POT) from lat/lon
# 2. Apply ``remove_radiation_zero_offset()`` to subtract the
#    nighttime bias from the whole series (``correct_nighttime_offset=True``)
# 3. Use SW_IN_POT as the daytime/nighttime divider (threshold 20 W/m2)
# 4. Set nighttime gaps to zero (physics)
# 5. Fill daytime gaps with XGBoost trained on daytime data only
# 6. Assemble the two parts into one gap-free series
#
# No additional driver variables are needed.  SW_IN_POT plus timestamp
# features and lag/rolling features of SW_IN_POT are sufficient — the
# potential-radiation curve already encodes solar angle, day length and
# seasonal amplitude, the dominant drivers of SW_IN variability.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

# %%
# Site configuration and data loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SITE_LAT = 47.286417  # CH-DAV Davos, Switzerland
SITE_LON = 7.733750
SITE_UTC_OFFSET = 1
TARGET_COL = 'Rg_f'  # Shortwave incoming radiation (W/m2)

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
# Run the gap-filler
# ^^^^^^^^^^^^^^^^^^
#
# This is the recommended configuration for SW_IN:
#
# - ``correct_nighttime_offset=True`` strips the sensor's nighttime bias
#   before any modelling, so the daytime model trains on physically
#   consistent values and the published series is exactly zero at night.
# - No ``context_df`` — only SW_IN_POT and timestamp features are used.

gf = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    nighttime_threshold=0.001,  # W/m2: SW_IN_POT < threshold -> night (matches remove_radiation_zero_offset)
    correct_nighttime_offset=True,
    reduce_features=False,
    verbose=1,
    # XGBoost hyperparameters
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
# Formatted report
# ^^^^^^^^^^^^^^^^
#
# ``report()`` prints parameters, data & performance, the flag distribution,
# and the daytime XGBoost scores.

gf.report()

# %%
# Inspect results programmatically
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

r = gf.results
gapfilled = r.gapfilled
flag = r.flag

print(f"\nResult columns: {list(r.gapfilling_df.columns)}")
print(f"  '{TARGET_COL}_offset_corrected' is the bias-corrected input series.")

if r.scores_traintest:
    print(f"\nDaytime model performance (train/test split):")
    print(f"  R2:   {r.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r.scores_traintest.get('rmse', float('nan')):.2f} W/m2")
    print(f"  MAE:  {r.scores_traintest.get('mae', float('nan')):.2f} W/m2")

# %%
# SHAP feature importances
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SW_IN_POT and its lag/rolling variants should dominate, followed by
# timestamp features (hour, DOY, sin/cos encodings).

if r.feature_importances is not None:
    fi = r.feature_importances.copy()
    print(f"\nTop 10 features by SHAP importance (daytime model):")
    print(fi.head(10).to_string())

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
axes[1].set_title('Gap-Filled SW_IN\n(offset-corrected + XGBoost)',
                  fontsize=11, fontweight='bold')

fig.suptitle('SW_IN Gap-Filling: Physics + XGBoost', fontsize=13, fontweight='bold')
plt.show()

print("SW_IN gap-filling example complete.")
