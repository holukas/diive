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
measurement that breaks it. It then sweeps a set of configurations and
scores each one on withheld daytime-gap RMSE computed at runtime.

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
# which is exactly what the timestamp cannot supply.
#
# Choosing a context_df source
# ----------------------------
#
# Preference order for a context_df radiation source, best first: a
# co-located second sensor (pyranometer or PPFD) that sees the same sky; a
# nearby station's radiation if the site is climatically similar; and, where
# no local or neighbouring sensor exists, satellite or reanalysis SW_IN such
# as ERA5-Land. All carry synoptic sky state the timestamp cannot, so even a
# coarse reanalysis product beats the timestamp-only climatology.
#
# The rest of the example runs a sweep of configurations and scores every
# one on withheld daytime-gap RMSE, so the numbers below come from the run,
# not from the text.

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
# ``context_df`` we will feed the model. In 2020 it is nearly gap-free.

SITE_LAT = 47.286417  # CH-DAV Davos, Switzerland
SITE_LON = 7.733750
SITE_UTC_OFFSET = 1
TARGET_COL = 'Rg_f'      # Shortwave incoming radiation (W/m2)
CONTEXT_COL = 'PPFD'     # Second radiation sensor, drives the sky state

# Shared XGBoost hyperparameters so every config is comparable.
XGB_KWARGS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

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
print(f"Missing values in {CONTEXT_COL} (clean context): "
      f"{df[CONTEXT_COL].isnull().sum()}")

# A deliberately gappy version of the context sensor: punch ~40% additional
# random gaps into PPFD to mimic a second sensor that is itself incomplete.
rng_ctx = np.random.default_rng(seed=1)
ppfd_gappy = df[CONTEXT_COL].copy()
ctx_observed = ppfd_gappy.dropna().index
ctx_gap_idx = rng_ctx.choice(ctx_observed, size=int(0.40 * len(ctx_observed)),
                             replace=False)
ppfd_gappy.loc[ctx_gap_idx] = np.nan
df_ctx_gappy = ppfd_gappy.to_frame()
print(f"Missing values in {CONTEXT_COL} (gappy context): "
      f"{df_ctx_gappy[CONTEXT_COL].isnull().sum()} "
      f"({100 * df_ctx_gappy[CONTEXT_COL].isnull().mean():.1f}%)")


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
# A helper that builds, runs and scores one configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Every config shares ``XGB_KWARGS`` and the site coordinates, so only the
# knobs under test differ. The helper returns the daytime-gap RMSE, the
# held-out R2, the number of features the daytime model used, and the raw
# results object for later inspection.

def run_config(label, **kwargs):
    """Build, run and score one SWINGapFillerXGBoost configuration."""
    gf = dv.gapfilling.SWINGapFillerXGBoost(
        series=df[TARGET_COL],
        lat=SITE_LAT,
        lon=SITE_LON,
        utc_offset=SITE_UTC_OFFSET,
        verbose=1,
        **kwargs,
        **XGB_KWARGS,
    )
    gf.run()
    r = gf.results
    rmse, n_gaps = daytime_gap_rmse(r)
    r2 = r.scores_traintest.get('r2', float('nan')) if r.scores_traintest else float('nan')
    # Number of features the daytime model actually used. feature_importances
    # has one row per feature; accepted_features is only populated when SHAP
    # reduction runs.
    if r.feature_importances is not None:
        n_features = int(len(r.feature_importances))
    elif r.accepted_features is not None:
        n_features = int(len(r.accepted_features))
    else:
        n_features = 0
    print(f"[{label}] daytime-gap RMSE = {rmse:.1f} W/m2, "
          f"held-out R2 = {r2:.3f}, features = {n_features}, "
          f"scored over {n_gaps} records")
    return dict(label=label, rmse=rmse, r2=r2, n_features=n_features,
                results=r)


# %%
# The sweep
# ^^^^^^^^^
#
# Seven configurations, grouped in three families:
#
# * no context (the climatology ceiling, plus two variations that cannot lift
#   it: short-gap interpolation and SHAP feature reduction);
# * a clean, near-complete second sensor (breaks the ceiling; interpolation
#   now hurts because it overwrites good model fills on short gaps);
# * a gappy second sensor (still helps, but some records fall back; here
#   interpolation earns its keep again on the short gaps the context missed).
#
# Each ``run_config`` call fits its own XGBoost model, so the block runs
# several fits and takes a couple of minutes.

runs = []

# Family 1: no context -- everything is a function of the timestamp.
runs.append(run_config(
    "1 no context (ceiling)"))
runs.append(run_config(
    "2 no context + interp=2",
    interpolate_short_gaps=2))
runs.append(run_config(
    "3 no context + reduce_features",
    reduce_features=True))

# Family 2: a clean, near-complete second radiation sensor.
runs.append(run_config(
    "4 clean PPFD context",
    context_df=df[[CONTEXT_COL]]))
runs.append(run_config(
    "5 clean PPFD context + interp=2",
    context_df=df[[CONTEXT_COL]],
    interpolate_short_gaps=2))

# Family 3: a gappy second sensor (~40% extra gaps punched into PPFD).
runs.append(run_config(
    "6 gappy PPFD context",
    context_df=df_ctx_gappy))
runs.append(run_config(
    "7 gappy PPFD context + interp=2",
    context_df=df_ctx_gappy,
    interpolate_short_gaps=2))


# %%
# Comparison table
# ^^^^^^^^^^^^^^^^^
#
# All numbers come from this run: withheld daytime-gap RMSE (lower is
# better), the daytime model's held-out R2, and the feature count.

print("\n" + "=" * 68)
print("SWINGapFillerXGBoost configuration sweep")
print("=" * 68)
header = f"{'config':<34}{'RMSE W/m2':>11}{'held-R2':>9}{'n_feat':>8}"
print(header)
print("-" * 68)
for run in runs:
    print(f"{run['label']:<34}{run['rmse']:>11.1f}"
          f"{run['r2']:>9.3f}{run['n_features']:>8d}")
print("-" * 68)

rmse_ceiling = runs[0]['rmse']
rmse_context = runs[3]['rmse']
print(f"Second sensor vs. ceiling: "
      f"{100 * (1 - rmse_context / rmse_ceiling):.0f}% lower RMSE")


# %%
# Interpretation
# ^^^^^^^^^^^^^^
#
# Reading the table:
#
# * The second radiation sensor is the biggest lever by far. Configs 4-7 sit
#   well below the no-context configs 1-3: PPFD carries the cloudy-vs-clear
#   sky state that no timestamp feature can.
# * Short-gap interpolation helps only when the model is climatology-bound
#   (config 2 vs 1) or when the context sensor is gappy (config 7 vs 6). With
#   a strong, near-complete context sensor it *hurts* (config 5 vs 4), because
#   clearness-index interpolation overwrites model fills that were already
#   better on those short gaps. This is why ``interpolate_short_gaps`` is off
#   by default.
# * ``reduce_features`` (config 3) is near-neutral: dropping low-SHAP features
#   does not raise the ceiling, it only trims the feature list.
# * Offset correction is on throughout and is near-neutral for this target.
#
# In short: get a second radiation measurement into ``context_df``; only reach
# for interpolation when you are still under the ceiling or your context
# sensor has holes.


# %%
# Inspect the reference "ceiling broken" configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The clean-context run (config 4) is the reference result once the ceiling is
# broken. Its results object carries the gap-filled series, the day/night flag,
# the held-out scores and the SHAP importances. (For the full formatted console
# report, call ``.report()`` on the gap-filler instance itself.)

r = runs[3]['results']  # config 4, clean PPFD context
print(f"\nReference configuration: {runs[3]['label']}")
print(f"Result columns: {list(r.gapfilling_df.columns)}")

if r.scores_traintest:
    print(f"\nDaytime model performance (train/test split, held-out):")
    print(f"  R2:   {r.scores_traintest.get('r2', float('nan')):.3f}")
    print(f"  RMSE: {r.scores_traintest.get('rmse', float('nan')):.2f} W/m2")
    print(f"  MAE:  {r.scores_traintest.get('mae', float('nan')):.2f} W/m2")


# %%
# SHAP feature importances (clean context)
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With the context sensor available, PPFD and its rolling/EMA variants carry
# the sky state and should rank alongside SW_IN_POT, well above the pure
# timestamp features.

if r.feature_importances is not None:
    fi = r.feature_importances.copy()
    print(f"\nTop 10 features by SHAP importance (daytime model, clean context):")
    print(fi.head(10).to_string())


# %%
# Visualize: observed vs gap-filled heatmaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Three panels: the observed record with gaps, the no-context fill (still on
# the ceiling), and the clean-context fill (ceiling broken).

rmse_default = runs[0]['rmse']
rmse_ctx = runs[3]['rmse']
gapfilled_default = runs[0]['results'].gapfilled
gapfilled_context = runs[3]['results'].gapfilled

fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                         gridspec_kw={'wspace': 0.2},
                         constrained_layout=True)

dv.plotting.HeatmapDateTime(series=df[TARGET_COL]).plot(
    ax=axes[0], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[0].set_title('Observed SW_IN\n(with gaps)', fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gapfilled_default).plot(
    ax=axes[1], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[1].set_title(f'Gap-filled, no context\nRMSE {rmse_default:.0f} W/m2',
                  fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=gapfilled_context).plot(
    ax=axes[2], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[2].set_title(f'Gap-filled, PPFD context\nRMSE {rmse_ctx:.0f} W/m2',
                  fontsize=11, fontweight='bold')

fig.suptitle('SW_IN Gap-Filling: the second radiation sensor breaks the '
             'climatology ceiling', fontsize=13, fontweight='bold')
plt.show()

print("\nSW_IN gap-filling example complete.")
