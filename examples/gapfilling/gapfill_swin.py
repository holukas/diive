"""
=====================================
SW_IN Gap-Filling (Physics + XGBoost)
=====================================

Gap-fill shortwave incoming radiation with physics-aware partitioning
combined with XGBoost.

Nighttime gaps are set to zero (no solar radiation after sunset).
Daytime gaps are filled with XGBoost. The two parts are assembled into one
complete, physically consistent time series.

This example works through the one thing that matters most for SW_IN
gap-filling accuracy: the climatology ceiling, and the second radiation
measurement that breaks it.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Overview
# ^^^^^^^^
#
# Shortwave incoming radiation (SW_IN) has a hard physical constraint:
# it is exactly zero at night. SWINGapFillerXGBoost uses potential
# radiation (SW_IN_POT, computed from lat/lon) to split the record into
# daytime and nighttime, sets nighttime gaps to zero, and fills daytime
# gaps with an XGBoost model.
#
# The default configuration needs no extra driver variables. SW_IN_POT plus
# timestamp features already encode solar angle, day length and seasonal
# amplitude. But there is a ceiling to what that buys you, and this example
# is about where the ceiling is and how to get past it.
#
# The climatology ceiling
# -----------------------
#
# With no context drivers, every feature the model sees is a deterministic
# function of the timestamp: SW_IN_POT, the timestamp features and the
# record number are all fixed once the time is known. So the model can only
# reproduce a climatology, the expected SW_IN for that time of day and year.
# It cannot know whether a particular gap was overcast or clear, because
# nothing in its inputs carries the sky state. Adding more timestamp-derived
# features (lags, rolling means and EMAs of SW_IN_POT) does not help: they
# are the same function of the timestamp again.
#
# What breaks the ceiling is a *second radiation measurement* passed through
# ``context_df`` -- a co-located pyranometer, a PPFD sensor, or a nearby
# station. Unlike air temperature or VPD it measures the sky state directly,
# which is exactly what the timestamp cannot supply. This example compares
# the two configurations head to head.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

# %%
# Site configuration and data loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The example dataset carries both ``Rg_f`` (the shortwave incoming
# radiation we gap-fill) and ``PPFD`` (photosynthetic photon flux density),
# a second, independent radiation sensor at the same site. PPFD is the
# ``context_df`` we will feed the model.

SITE_LAT = 47.286417  # CH-DAV Davos, Switzerland
SITE_LON = 7.733750
SITE_UTC_OFFSET = 1
TARGET_COL = 'Rg_f'      # Shortwave incoming radiation (W/m2)
CONTEXT_COL = 'PPFD'     # Second radiation sensor, drives the sky state

df_orig = dv.load_exampledata_parquet()
df = df_orig.copy()
keep = df.index.year == 2020
df = df[keep].copy()

# Keep an untouched copy of the target so we can score the fill against the
# values we are about to remove.
truth = df[TARGET_COL].copy()

# Introduce artificial gaps: randomly remove 15% of observed values.
rng = np.random.default_rng(seed=42)
observed_idx = df[TARGET_COL].dropna().index
gap_idx = rng.choice(observed_idx, size=int(0.15 * len(observed_idx)), replace=False)
df.loc[gap_idx, TARGET_COL] = np.nan

print(f"Data loaded: {len(df)} records")
print(f"Missing values in {TARGET_COL}: {df[TARGET_COL].isnull().sum()} "
      f"({100 * df[TARGET_COL].isnull().mean():.1f}%)")


# %%
# A small scorer for the withheld gaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We know the true SW_IN at every artificial gap, so we can measure the fill
# directly. Nighttime gaps are trivially zero, so the interesting comparison
# is at *daytime* gaps -- that is where the model has to work.

def daytime_gap_rmse(result):
    """RMSE of the fill against truth, at artificial daytime gap records."""
    swinpot = result.gapfilling_df['SW_IN_POT']
    daytime = swinpot >= 0.001
    idx = df.index.isin(gap_idx) & daytime.values
    err = result.gapfilled[idx] - truth[idx]
    return float(np.sqrt((err ** 2).mean())), int(idx.sum())


# %%
# Run 1: default configuration (the climatology ceiling)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# No ``context_df``. The model has only SW_IN_POT and timestamp features, so
# it fills every daytime gap with the expected clear-vs-cloudy-averaged value
# for that time of day and year. Note the default ``nighttime_threshold`` is
# 0.001 W/m2 (matching ``remove_nighttime_zero_offset``), not 20.

gf_default = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
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
gf_default.run()

rmse_default, n_gaps = daytime_gap_rmse(gf_default.results)
print(f"\nDefault (no context): daytime-gap RMSE = {rmse_default:.1f} W/m2 "
      f"over {n_gaps} records")


# %%
# Run 2: still no context, but interpolate short gaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The model never sees the target's own neighbours -- the feature engineer
# excludes the target from every feature -- so a short gap is exactly the
# case a timestamp-only model is blind to. ``interpolate_short_gaps=2``
# (1 h on 30-min data) fills gaps of one or two records by interpolating the
# clearness index (SW_IN / SW_IN_POT), which does use those neighbours. It
# never bridges a night. This is the lever that helps while you are still
# under the climatology ceiling.

gf_interp = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    interpolate_short_gaps=2,
    verbose=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
gf_interp.run()

rmse_interp, _ = daytime_gap_rmse(gf_interp.results)
print(f"\nNo context + short-gap interpolation: daytime-gap RMSE = "
      f"{rmse_interp:.1f} W/m2 over {n_gaps} records")


# %%
# Run 3: a second radiation sensor (breaking the ceiling)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The only change from Run 1 is ``context_df=df[[CONTEXT_COL]]``. PPFD tracks
# the same sky the pyranometer sees, so where a gap was cloudy PPFD is low and
# the model follows it down instead of averaging over the climatology. This is
# a different, larger lever than interpolation: it removes the ceiling itself
# rather than patching short gaps under it.
#
# The context sensor does not need to be in W/m2 or gap-free: the model learns
# the relationship from whatever overlap exists, and records where it is
# missing fall back to the climatology fill. Here PPFD is nearly complete, so
# the model resolves short gaps well on its own -- with a strong, near-complete
# second sensor, ``interpolate_short_gaps`` adds little and can even replace a
# good model fill with a worse interpolation. Interpolation earns its keep
# under the ceiling (Run 2) or when the context sensor is itself gappy.

gf_context = dv.gapfilling.SWINGapFillerXGBoost(
    series=df[TARGET_COL],
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=SITE_UTC_OFFSET,
    context_df=df[[CONTEXT_COL]],
    verbose=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
gf_context.run()

rmse_context, _ = daytime_gap_rmse(gf_context.results)
print(f"\nWith PPFD context:    daytime-gap RMSE = {rmse_context:.1f} W/m2 "
      f"over {n_gaps} records")
print(f"Improvement from the second sensor vs. the default: "
      f"{100 * (1 - rmse_context / rmse_default):.0f}%")


# %%
# Formatted report
# ^^^^^^^^^^^^^^^^^
#
# ``report()`` prints parameters, data & performance, the flag distribution,
# and the daytime XGBoost scores. The flag distribution separates model fills
# (1) from fallback fills (2), where the context sensor was missing.

gf_context.report()


# %%
# Inspect results programmatically
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

r = gf_context.results
print(f"\nResult columns: {list(r.gapfilling_df.columns)}")

if r.scores_traintest:
    print(f"\nDaytime model performance (train/test split, held-out):")
    print(f"  R2:   {r.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r.scores_traintest.get('rmse', float('nan')):.2f} W/m2")
    print(f"  MAE:  {r.scores_traintest.get('mae', float('nan')):.2f} W/m2")


# %%
# SHAP feature importances
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With the context sensor available, PPFD and its rolling/EMA variants carry
# the sky state and should rank alongside SW_IN_POT, well above the pure
# timestamp features.

if r.feature_importances is not None:
    fi = r.feature_importances.copy()
    print(f"\nTop 10 features by SHAP importance (daytime model, with context):")
    print(fi.head(10).to_string())


# %%
# Visualize: observed vs gap-filled heatmaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                         gridspec_kw={'wspace': 0.2},
                         constrained_layout=True)

dv.plotting.HeatmapDateTime(series=df[TARGET_COL]).plot(
    ax=axes[0], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[0].set_title('Observed SW_IN\n(with gaps)', fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gf_default.results.gapfilled).plot(
    ax=axes[1], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[1].set_title(f'Gap-filled, no context\nRMSE {rmse_default:.0f} W/m2',
                  fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gf_context.results.gapfilled).plot(
    ax=axes[2], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[2].set_title(f'Gap-filled, PPFD context\nRMSE {rmse_context:.0f} W/m2',
                  fontsize=11, fontweight='bold')

fig.suptitle('SW_IN Gap-Filling: the second radiation sensor breaks the '
             'climatology ceiling', fontsize=13, fontweight='bold')
plt.show()

print("\nSW_IN gap-filling example complete.")
