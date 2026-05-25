"""
====================================================
Flux Processing Chain - Composable Functions
====================================================

The flux processing chain is also exposed as standalone pure functions, one per
level.  Each function takes a ``FluxLevelData`` container and returns a new
one — no shared state, no orchestrator class required.

This example runs the **full L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1** pipeline
using composable functions and demonstrates the key advantages of the functional
API: partial pipelines, custom steps, and **branching** (running three gap-filling
methods — Random Forest, XGBoost, and MDS — from the same L3.3 state without
re-doing any upstream work).

Compare with ``fluxprocessingchain.py`` for the same chain via the
``FluxProcessingChain`` orchestrator class.
"""

# %%
# Imports
# ^^^^^^^

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.flux.fluxprocessingchain import (
    init_flux_data,
    make_level32_detector,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)

# %%
# Load data
# ^^^^^^^^^

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.loc['2024-06':'2024-06']  # one month for speed

# %%
# Step 1: build the initial FluxLevelData container
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``init_flux_data`` adds potential radiation and day/night flags, assembles
# the frozen site-metadata record, and returns a container ready for the
# first level.

data = init_flux_data(
    df=df,
    fluxcol="FC",
    site_lat=47.41887,        # CH-HON
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)
print(data)   # FluxLevelData has a useful __repr__

# %%
# Step 2: run Level-2 (quality flag expansion)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each test is enabled by passing a config dict containing at least
# ``{'apply': True}``.  Pass ``None`` (or omit) to skip a test.

data = run_level2(
    data,
    ssitc={'apply': True, 'setflag_timeperiod': None},
    gas_completeness={'apply': True},
    spectral_correction_factor={'apply': True},
    signal_strength={
        'apply': True,
        'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
        'method': 'discard below',
        'threshold': 60,
    },
    raw_data_screening_vm97={
        'apply': True,
        'spikes': True, 'amplitude': False, 'dropout': True,
        'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
        'discont_hf': False, 'discont_sf': False,
    },
)
print(f"After L2: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# Step 3: run Level-3.1 (storage correction)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

data = run_level31(data, gapfill_storage_term=True, set_storage_to_zero=False)
print(f"After L3.1: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# Step 4: run Level-3.2 (outlier removal)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Level-3.2 uses a stateful ``StepwiseOutlierDetection`` instance.  The
# ``make_level32_detector`` factory wires it to the right ``dfin`` / ``col``
# / site coordinates so you don't have to.  Call any number of
# ``flag_outliers_*`` / ``addflag`` methods on it, then hand it to
# ``run_level32``.

sod = make_level32_detector(data)
sod.flag_outliers_hampel_test(
    window_length=48 * 13,
    n_sigma_daytime=5.5, n_sigma_nighttime=5.5,
    use_differencing=True, separate_daytime_nighttime=True,
    showplot=False, verbose=True, repeat=True,
)
sod.addflag()

data = run_level32(data, outlier_detector=sod)
print(f"After L3.2: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# QCF heatmaps (L3.2 quality overview)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each level's QCF object can show diagnostic heatmaps — a date-vs-time grid
# coloured by QCF value (0=good, 1=soft warning, 2=hard failure).  Useful for
# spotting instrument outages or seasonal quality patterns.

data.levels.level32_qcf.showplot_qcf_heatmaps()

# %%
# Step 5: run Level-3.3 (USTAR turbulence filtering)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Remove low-turbulence nighttime periods using a constant USTAR threshold.
# Applied only to CO2/CH4/N2O — not to energy fluxes (H, LE).
#
# Pass the 16th, 50th, and 84th percentiles of a bootstrap USTAR threshold
# distribution (from REddyProc or hesseflux) to quantify the uncertainty.
# Here we use one scenario for brevity.
#
# After this call ``data.filteredseries`` is set to ``None`` because there
# is no single unambiguous filtered series when multiple USTAR scenarios
# exist.  Always access per-scenario series explicitly.

data = run_level33_constant_ustar(
    data,
    thresholds=[0.30],           # m s-1; site-specific, typically CUT_50
    threshold_labels=['CUT_50'],
    showplot=False,
    verbose=True,
)

# data.filteredseries is None here — use the scenario dict instead
flux_l33 = data.levels.filteredseries_level33_qcf['CUT_50']
print(f"After L3.3 (CUT_50): {flux_l33.dropna().count()} valid records")

# %%
# Step 6: feature engineering for gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Build a ``FeatureEngineer`` with the meteorological driver columns that will
# serve as gap-filling predictors.  ``target_col`` is a required placeholder —
# its value does not matter as long as it is not in the feature list.
#
# All feature columns must exist in ``data.full_df`` (the original input
# dataframe), not ``data.fpc_df``.

FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

engineer = FeatureEngineer(
    target_col='_target_',  # placeholder; value irrelevant for L4.1
    features_lag=[-2, -1],
    features_lag_stepsize=1,
    features_lag_exclude_cols=None,
    features_rolling=[2, 4, 12, 24, 48],
    features_rolling_exclude_cols=None,
    features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
    features_diff=[1, 2],
    features_diff_exclude_cols=None,
    features_ema=[6, 12, 24, 48],
    features_ema_exclude_cols=None,
    features_poly_degree=2,
    features_poly_exclude_cols=None,
    features_stl=True,
    features_stl_method='stl',
    features_stl_seasonal_period=48,
    features_stl_exclude_cols=None,
    features_stl_components=['trend', 'seasonal', 'residual'],
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    sanitize_timestamp=True,
    verbose=1,
)

# %%
# Step 7a: gap-fill with Random Forest
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``run_level41_rf`` runs one RF model per USTAR scenario found in L3.3.
# Feature engineering (``engineer.fit_transform``) runs once and is reused
# across all USTAR scenarios — no redundant computation.

data = run_level41_rf(
    data,
    features=FEATURES,
    engineer=engineer,
    reduce_features=True,
    verbose=1,
    # RF hyperparameters — use n_estimators=350, max_depth=15 in production
    n_estimators=2,
    max_depth=1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
print("Random Forest gap-filling complete")

# %%
# Step 7b: gap-fill with XGBoost (branch from same L3.3 state)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The composable API makes branching trivial: pass the same ``data`` object
# (which already contains the L3.3 results and the RF outputs) directly to
# ``run_level41_xgb``.  No upstream work is repeated; only the XGBoost model
# is added.

data = run_level41_xgb(
    data,
    features=FEATURES,
    engineer=engineer,
    reduce_features=True,
    verbose=1,
    # XGBoost hyperparameters — use n_estimators=350, max_depth=6 in production
    n_estimators=2,
    max_depth=1,
    learning_rate=0.05,
    early_stopping_rounds=30,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
)
print("XGBoost gap-filling complete")

# %%
# Step 7c: gap-fill with MDS (branch from same L3.3 state)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Marginal Data Substitution requires no model training: it searches a
# look-up table of meteorologically similar half-hours to fill each gap.
# No ``FeatureEngineer`` is needed.
#
# Driver columns must exist in ``data.full_df``.  MDS expects VPD in **kPa**;
# EddyPro outputs VPD in hPa — divide by 10 if the raw column is used.
# ``run_level41_mds`` raises a ``UserWarning`` automatically when the median
# VPD looks like hPa (> 10).

data = run_level41_mds(
    data,
    swin='SW_IN_T1_47_1_gfXG',
    ta='TA_T1_47_1_gfXG',
    vpd='VPD_T1_47_1_gfXG',   # kPa; divide by 10 if your column is in hPa
    ta_tol=2.5,
    vpd_tol=0.5,
)
print("MDS gap-filling complete")

# %%
# Step 8: inspect typed per-level results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.levels`` is a typed ``LevelResults`` dataclass.  Every per-level
# object lives behind a named attribute — no magic-string dict lookups.

print(f"Levels run: {data.level_ids}")
print(f"L3.3 instance:   {type(data.levels.level33).__name__}")
print(f"L3.3 QCF keys:   {list(data.levels.level33_qcf.keys())}")
print(f"L4.1 RF keys:    {list(data.levels.level41_rf.keys())}")
print(f"L4.1 XGB keys:   {list(data.levels.level41_xgb.keys())}")
print(f"L4.1 MDS keys:   {list(data.levels.level41_mds.keys())}")

# %%
# Step 9: find gap-filled column names
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.gapfilled_cols()`` returns a nested dict
# ``{method: {ustar_scenario: column_name}}`` so you don't have to dig into
# the model instances to find the right column in ``data.fpc_df``.

cols = data.gapfilled_cols()
print(f"Gap-filled columns: {cols}")

rf_col  = cols['rf']['CUT_50']
xgb_col = cols['xgb']['CUT_50']
mds_col = cols['mds']['CUT_50']

rf_gapfilled  = data.fpc_df[rf_col]
xgb_gapfilled = data.fpc_df[xgb_col]
mds_gapfilled = data.fpc_df[mds_col]

print(f"RF  gap-filled: {rf_gapfilled.dropna().count()} records in column '{rf_col}'")
print(f"XGB gap-filled: {xgb_gapfilled.dropna().count()} records in column '{xgb_col}'")
print(f"MDS gap-filled: {mds_gapfilled.dropna().count()} records in column '{mds_col}'")

# %%
# Step 10: data-availability summary across all levels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.summary()`` prints per-level valid record counts split by daytime
# and nighttime.  Useful for a quick sanity check before exporting results.

print(data.summary())

# %%
# Step 11: model performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Access the trained model instance for each USTAR scenario and gap-filling
# method via ``data.levels.level41_rf[ustar_scen]``.  The instance exposes
# per-year R² scores and a convenience plot method.

rf_model  = data.levels.level41_rf['CUT_50']
xgb_model = data.levels.level41_xgb['CUT_50']
mds_model = data.levels.level41_mds['CUT_50']

rf_r2  = list(rf_model.scores_.values())[0]['r2']  if rf_model.scores_  else None
xgb_r2 = list(xgb_model.scores_.values())[0]['r2'] if xgb_model.scores_ else None

if rf_r2 is not None:
    print(f"Random Forest  R2: {rf_r2:.3f}")
if xgb_r2 is not None:
    print(f"XGBoost        R2: {xgb_r2:.3f}")

# Feature importance plots (ML methods only)
rf_model.showplot_feature_ranks_per_year(title="RF feature importance — NEE CUT_50")
xgb_model.showplot_feature_ranks_per_year(title="XGBoost feature importance — NEE CUT_50")

# MDS diagnostic plot
mds_model.showplot()

# %%
# Step 12: heatmap of gap-filled flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``HeatmapDateTime`` arranges the time series on a date x time-of-day grid.
# NaN cells are highlighted so data gaps remain visible even after filling.
# Plot both the RF and XGB gap-filled series side-by-side for comparison.

heatmap_rf = dv.plotting.HeatmapDateTime(series=rf_gapfilled)
heatmap_rf.plot(title=f"RF gap-filled flux  ({rf_col})")

heatmap_xgb = dv.plotting.HeatmapDateTime(series=xgb_gapfilled)
heatmap_xgb.plot(title=f"XGB gap-filled flux  ({xgb_col})")

heatmap_mds = dv.plotting.HeatmapDateTime(series=mds_gapfilled)
heatmap_mds.plot(title=f"MDS gap-filled flux  ({mds_col})")

# %%
# Step 13: cumulative gap-filled flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``CumulativeYear`` shows yearly cumulative sums day-by-day.  With one month
# of data the curve covers only June; with a full year it produces the classic
# annual NEE fingerprint used to report carbon-budget balance.

cumulative_rf = dv.plotting.CumulativeYear(
    series=rf_gapfilled,
    series_units="umol m-2 s-1",
)
cumulative_rf.plot()

cumulative_xgb = dv.plotting.CumulativeYear(
    series=xgb_gapfilled,
    series_units="umol m-2 s-1",
)
cumulative_xgb.plot()

cumulative_mds = dv.plotting.CumulativeYear(
    series=mds_gapfilled,
    series_units="umol m-2 s-1",
)
cumulative_mds.plot()

# %%
# Step 14: export the final dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.fpc_df`` holds every flag, QCF, and gap-filled column appended by
# the chain.  Export it directly or subset the columns you need.

final_df = data.fpc_df
print(f"Final dataframe: {final_df.shape[0]} rows x {final_df.shape[1]} cols")

# %%
# Why use the composable API?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - **Partial pipelines**: stop after any level — no chain overhead.
# - **Custom steps**: skip ``run_level32`` and write your own outlier logic on
#   ``data.fpc_df``, then continue with ``run_level33_constant_ustar``.
# - **Branching**: run three L4.1 methods (RF, XGBoost, MDS) from the same
#   L3.3 state by reusing the same ``FluxLevelData`` — no upstream re-runs.
# - **Pure functions**: each call returns a new container, never mutates the
#   input — easy to unit-test, easy to reason about.
#
# For the full L2-to-L4.1 chain (RF + XGBoost gap-filling), the
# ``FluxProcessingChain`` orchestrator is more concise — see
# ``fluxprocessingchain.py``.
