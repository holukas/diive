"""
====================================================
Flux Processing Chain - Composable Functions
====================================================

The flux processing chain is also exposed as standalone pure functions, one per
level.  Each function takes a ``FluxLevelData`` container and returns a new
one — no shared state, no orchestrator class required.

This example runs the **full L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1** pipeline and
demonstrates the key advantage of the composable functional API: **branching** — running
three gap-filling methods (Random Forest, XGBoost, MDS) from the same L3.3 state
without repeating any upstream work.

Swiss FluxNet workflow overview
--------------------------------
- **L2**   — Expand EddyPro quality flags; compute an overall Quality Control
             Flag (QCF) per half-hour.  Remove records with hard failures.
- **L3.1** — Add the single-point canopy-air-space CO2 storage term to the
             turbulent flux.  Without this, nighttime NEE is systematically
             under-estimated.
- **L3.2** — Remove remaining outliers (spikes, instrument glitches) that
             passed the L2 statistical tests.
- **L3.3** — Discard nighttime records measured under low-turbulence conditions
             (USTAR below a site-specific threshold).  Under stable stratification,
             CO2 drains laterally and the eddy-covariance method under-estimates
             ecosystem respiration.
- **L4.1** — Fill the data gaps created by L2-L3.3 with meteorology-driven
             models.  Required to compute annual carbon budgets.
"""

# %%
# Imports
# ^^^^^^^

import warnings

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.flux.fluxprocessingchain import (
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
# Step 1: initialise the FluxLevelData container
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``init_flux_data`` calculates potential radiation, derives daytime/nighttime
# flags, and assembles a frozen site-metadata record.  All downstream level
# functions read site coordinates and QCF thresholds from this metadata.
#
# ``daytime_accept_qcf_below=2`` keeps QCF=0 (all tests pass) and QCF=1 (soft
# warnings) for daytime records.  Set to 1 to accept only the strictest quality.

data = init_flux_data(
    df=df,
    fluxcol="FC",
    site_lat=47.41887,  # CH-HON
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,  # W m-2; SW_IN below this = nighttime
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)
print(data)  # FluxLevelData has a useful __repr__

# %%
# Step 2: Level-2 — quality flag expansion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Seven configurable tests expand the raw EddyPro quality indicators into
# explicit per-half-hour flags.  Each test is enabled by passing a config dict
# with ``{'apply': True, ...}``.  Omit or pass ``None`` to skip a test.
#
# Tests applied here:
#   - SSITC  : steady-state and integral turbulence characteristics (Foken 2004)
#   - Gas completeness  : fraction of raw 10/20 Hz records available
#   - Spectral correction factor  : overcorrected spectra are unphysical
#   - Signal strength   : low IRGA signal = dirty or wet optics
#   - VM97 spikes/dropout : raw-data spike and dropout detection (Vickers & Mahrt 1997)

data = run_level2(
    data,
    ssitc={'apply': True, 'setflag_timeperiod': None},
    gas_completeness={'apply': True},
    spectral_correction_factor={'apply': True},
    signal_strength={
        'apply': True,
        'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
        'method': 'discard below',  # low signal = problem on LI-7200
        'threshold': 60,
    },
    raw_data_screening_vm97={
        'apply': True,
        'spikes': True, 'amplitude': False, 'dropout': True,
        'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
        'discont_hf': False, 'discont_sf': False,
    },
)
print(f"After L2: {data.filteredseries.dropna().count()} valid records "
      f"(col: {data.filteredseries.name})")

# %%
# QCF heatmaps (L2 quality overview)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Shows which half-hours failed which tests on a date x time-of-day grid.
# Useful for spotting instrument outages, sensor contamination events, or
# persistent quality problems at certain times of day.

data.levels.level2_qcf.showplot_qcf_heatmaps()

# %%
# Step 3: Level-3.1 — storage correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The eddy-covariance flux FC measures only the turbulent vertical transport.
# During transitional periods (sunrise, sunset, after rain) CO2 accumulates in
# or drains from the canopy air space.  The storage flux SC_SINGLE corrects for
# this, giving the net ecosystem-atmosphere exchange:
#
#   NEE = FC + SC_SINGLE
#
# Gap-filling the storage term with a rolling median (``gapfill_storage_term=True``)
# prevents the correction from introducing additional gaps.

data = run_level31(data, gapfill_storage_term=True, set_storage_to_zero=False)
print(f"After L3.1: {data.filteredseries.dropna().count()} valid records "
      f"(col: {data.filteredseries.name})")

# %%
# Step 4: Level-3.2 — outlier removal
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Even after L2 flagging, spikes from transient sensor contamination, insects
# in the optical path, or brief electrical interference remain in the NEE record.
# The Hampel filter (median-absolute-deviation based, applied separately to
# daytime and nighttime) is robust to the asymmetric distribution of NEE.
#
# ``make_level32_detector`` wires the detector to the correct input column and
# site coordinates so you don't have to.  Chain additional ``flag_outliers_*``
# / ``addflag()`` calls before ``run_level32`` if needed.

sod = make_level32_detector(data)
sod.flag_outliers_hampel_test(
    window_length=48 * 13,  # 13-day rolling window (±6.5 days)
    n_sigma_daytime=5.5,
    n_sigma_nighttime=5.5,
    use_differencing=True,  # more sensitive to isolated spikes
    separate_daytime_nighttime=True,
    showplot=False,
    verbose=True,
    repeat=True,
)
sod.addflag()

data = run_level32(data, outlier_detector=sod)
print(f"After L3.2: {data.filteredseries.dropna().count()} valid records "
      f"(col: {data.filteredseries.name})")

# %%
# QCF heatmaps (L3.2 quality overview)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

data.levels.level32_qcf.showplot_qcf_heatmaps()

# %%
# Step 5: Level-3.3 — USTAR turbulence filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Under calm, stable nighttime conditions the atmospheric boundary layer
# decouples from the canopy.  Eddy-covariance underestimates ecosystem
# respiration because CO2 drains laterally rather than vertically.  The
# standard correction (Papale et al. 2006) removes nighttime records where
# the friction velocity USTAR falls below a site-specific threshold.
#
# The threshold is uncertain: derive its distribution externally (e.g. REddyProc)
# and pass the 16th, 50th, and 84th percentiles to quantify sensitivity.
# Here we use one scenario for brevity.
#
# USTAR filtering applies ONLY to CO2, CH4, and N2O fluxes.
# For energy fluxes (H, LE) use thresholds=[0], threshold_labels=['CUT_NONE'].
#
# After this call ``data.filteredseries`` is None — always access results via
# the per-scenario dict (see below).

data = run_level33_constant_ustar(
    data,
    thresholds=[0.30],  # m s-1 — site-specific value
    threshold_labels=['CUT_50'],  # label matches the bootstrap percentile
    showplot=False,
    verbose=True,
)

# Access per-scenario results explicitly
flux_l33 = data.levels.filteredseries_level33_qcf['CUT_50']
flux_l33_hq = data.levels.filteredseries_level33_hq['CUT_50']  # QCF=0 only

print(f"After L3.3 (CUT_50): {flux_l33.dropna().count()} accepted  |  "
      f"{flux_l33_hq.dropna().count()} high-quality (QCF=0)")

# %%
# Step 5b: gap analysis — structure of missing data before gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This is the right moment to inspect gaps: every quality-screening step has
# run and the series is exactly what the gap-filling models will receive.
#
# ``data.gap_stats()`` is a convenience method on the FluxLevelData container.
# It works after any level ('L2', 'L31', 'L32', 'L33') and returns a
# {scenario: GapStats} dict.  It is not part of any official level function —
# call it whenever you need a gap audit.

for scen, gs in data.gap_stats('L33').items():
    gs.report()  # Rich console report
    gs.showfig(title=f"Gap analysis before gap-filling  --  {scen}")

# %%
# Step 6: feature engineering for ML gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ML gap-filling models require a rich feature set that captures the diurnal
# and seasonal drivers of NEE (radiation, temperature, VPD) plus their
# temporal context (lags, rolling statistics, trends).  The 8-stage
# ``FeatureEngineer`` pipeline creates these automatically.
#
# All feature columns must exist in ``data.full_df`` (the original input
# dataframe), not ``data.fpc_df``.
#
# ``target_col='_target_'`` is a required placeholder; its value is irrelevant
# for L4.1 because the engineer is applied to predictor columns only.
#
# The same engineer instance is passed to both RF and XGBoost — feature
# engineering runs once and is reused across all USTAR scenarios and methods.

FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

engineer = FeatureEngineer(
    target_col='_target_',  # placeholder; irrelevant for L4.1
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
# An ensemble of decision trees, each trained on a bootstrapped subset of the
# data.  Robust to outliers, interpretable via SHAP feature importances.
# Production settings: n_estimators >= 350, max_depth >= 15.

data = run_level41_rf(
    data,
    features=FEATURES,
    engineer=engineer,  # same instance reused by XGBoost below
    reduce_features=True,  # SHAP-based feature selection
    verbose=1,
    # Demo settings — use n_estimators=350, max_depth=15 in production
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
# Gradient boosted trees.  Generally captures stronger non-linear patterns than
# RF and trains faster at equal n_estimators.  The same ``engineer`` instance
# is passed; feature engineering is not repeated.
#
# This is the composable API's key advantage: pass the same ``data`` object to
# run a second method — no upstream work is repeated.

data = run_level41_xgb(
    data,
    features=FEATURES,
    engineer=engineer,
    reduce_features=True,
    verbose=1,
    # Demo settings — use n_estimators=350, max_depth=6 in production
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
# Marginal Data Substitution (Reichstein et al. 2005) fills gaps by searching
# a look-up table of meteorologically similar half-hours within a moving window.
# No model training needed; no risk of overfitting.
#
# Driver columns must exist in ``data.full_df``.
# MDS requires VPD in kPa — EddyPro outputs VPD in hPa, so divide by 10 if
# using the raw EddyPro column.  ``run_level41_mds`` issues a UserWarning
# automatically when the median VPD looks like hPa (> 10).

VPD_COL = 'VPD_T1_47_1_gfXG'
vpd_median = data.full_df[VPD_COL].median()
if vpd_median > 10:
    warnings.warn(
        f"VPD column '{VPD_COL}' has median {vpd_median:.1f} — "
        f"looks like hPa; MDS requires kPa.  Divide by 10.",
        UserWarning, stacklevel=1,
    )

data = run_level41_mds(
    data,
    swin='SW_IN_T1_47_1_gfXG',
    ta='TA_T1_47_1_gfXG',
    vpd=VPD_COL,  # kPa; divide by 10 if your column is in hPa
    ta_tol=2.5,  # Reichstein et al. 2005 default tolerances
    vpd_tol=0.5,
)
print("MDS gap-filling complete")

# %%
# Step 8: collect all gap-filling results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.levels.level41_methods()`` returns a nested dict keyed by method name
# and USTAR scenario.  Use this to iterate over methods uniformly rather than
# accessing level41_rf / level41_xgb / level41_mds individually.
#
# ``data.gapfilled_cols()`` returns the column names in ``fpc_df`` for each
# method and scenario — useful for plotting and export without digging into
# model instances.

all_models = data.levels.level41_methods()
# {'mds': {'CUT_50': <FluxMDS>},
#  'rf':  {'CUT_50': <LongTermGapFillingRandomForestTS>},
#  'xgb': {'CUT_50': <LongTermGapFillingXGBoostTS>}}

cols = data.gapfilled_cols()
# {'rf': {'CUT_50': 'NEE_..._gfRF'}, 'xgb': {'CUT_50': '..._gfXG'}, 'mds': {'CUT_50': '...'}}
print(f"Gap-filled columns:\n  {cols}")

rf_col = cols['rf']['CUT_50']
xgb_col = cols['xgb']['CUT_50']
mds_col = cols['mds']['CUT_50']

rf_gapfilled = data.fpc_df[rf_col]
xgb_gapfilled = data.fpc_df[xgb_col]
mds_gapfilled = data.fpc_df[mds_col]

# %%
# Step 9: gap-filling fraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``FLAG_*_ISFILLED`` column records the origin of each value:
#   0 = directly measured (passed all QC filters)
#   1 = gap-filled by the primary model
#   2 = gap-filled by the fallback model (if primary could not predict)
#
# Reporting this fraction is standard in flux papers.

for method_key, scen_cols in cols.items():
    for scen, gf_col in scen_cols.items():
        flag_col = f"FLAG_{gf_col}_ISFILLED"
        if flag_col in data.fpc_df.columns:
            flags = data.fpc_df[flag_col]
            n_measured = int((flags == 0).sum())
            n_filled = int((flags == 1).sum())
            total = n_measured + n_filled
            print(f"{method_key} {scen}: {n_measured}/{total} measured "
                  f"({100 * n_measured / max(total, 1):.1f}%), "
                  f"{n_filled} gap-filled ({100 * n_filled / max(total, 1):.1f}%)")

# %%
# Step 10: data-availability summary across all levels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print(data.summary())

# %%
# Step 11: model performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Access trained model instances and their per-year R² scores.
# ML methods expose ``showplot_feature_ranks_per_year()``; MDS exposes
# ``showplot()`` which shows gap-filling coverage by meteorological window.

rf_model = data.levels.level41_rf['CUT_50']
xgb_model = data.levels.level41_xgb['CUT_50']
mds_model = data.levels.level41_mds['CUT_50']

rf_r2 = list(rf_model.scores_.values())[0]['r2'] if rf_model.scores_ else None
xgb_r2 = list(xgb_model.scores_.values())[0]['r2'] if xgb_model.scores_ else None

if rf_r2 is not None:
    print(f"Random Forest  R2: {rf_r2:.3f}")
if xgb_r2 is not None:
    print(f"XGBoost        R2: {xgb_r2:.3f}")

rf_model.showplot_feature_ranks_per_year(title="RF feature importance — NEE CUT_50")
xgb_model.showplot_feature_ranks_per_year(title="XGBoost feature importance — NEE CUT_50")
mds_model.showplot()

# %%
# Step 12: heatmap of gap-filled flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.plot_gapfilled_heatmaps()`` shows the measured flux (before gap-filling)
# in the top panel, followed by one panel per gap-filling method that has been run.
# All panels share the same colour scale so structural differences between methods
# are immediately comparable.  Works with any subset of methods.

data.plot_gapfilled_heatmaps(
    ustar_scenario='CUT_50',
    units='umol m-2 s-1',
)

# %%
# Step 13: cumulative gap-filled NEE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Cumulative NEE (daily running sum) is the standard diagnostic for the annual
# carbon budget.  The sign convention follows eddy covariance: negative = carbon
# uptake by the ecosystem, positive = carbon release.
#
# Unit conversion from umol CO2 m-2 s-1 to gC m-2:
#   gC per timestep = umol CO2 m-2 s-1  x  12.011 g mol-1  x  1e-6 mol umol-1
#                     x  1800 s per 30-min timestep
# Summing over the full time series gives the cumulative carbon balance in gC m-2.
#
# With one month of data the curves cover June only; over a full year they
# produce the classic NEE fingerprint used in carbon-budget reporting.

UMOL_TO_GC = 12.011 * 1e-6 * 1800  # conversion factor for 30-min data

# ``data.plot_cumulative_comparison()`` overlays RF, XGBoost, and MDS on a
# single axes for the same USTAR scenario so structural differences between
# methods are immediately visible.  The dashed grey line shows the measured-only
# series (gaps contribute zero), making the magnitude of the gap-filling
# correction transparent.  Works with any subset of methods — only the ones
# that have been run appear in the plot.

data.plot_cumulative_comparison(
    ustar_scenario='CUT_50',
    conv_factor=UMOL_TO_GC,
    units='gC m-2',
)

# %%
# Step 14: export the final dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.fpc_df`` holds every flag, QCF, storage-correction, gap-filled column
# and their ISFILLED flags accumulated by the chain.  Export it directly or
# select the columns relevant to your analysis.
#
# Typical columns to keep for publication:
#   - the gap-filled NEE column  (e.g. NEE_L3.1_L3.3_CUT_50_QCF_gfRF)
#   - its ISFILLED flag          (distinguishes measured from modelled)
#   - the L2 QCF                 (FLAG_L2_FC_QCF)
#   - USTAR                      (already in fpc_df from init)
#
# **Next step — flux partitioning:**
# Gap-filled NEE can be partitioned into gross primary production (GPP) and
# ecosystem respiration (Reco) using nighttime or daytime approaches.
# Use an external tool such as REddyProc (R) for this step.

final_df = data.fpc_df
print(f"Final dataframe: {final_df.shape[0]} rows x {final_df.shape[1]} cols")

# %%
# Why use the composable API?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - **Partial pipelines**: stop after any level — no chain overhead.
# - **Custom steps**: replace ``run_level32`` with your own outlier logic on
#   ``data.fpc_df``, then continue with ``run_level33_constant_ustar``.
# - **Branching**: run RF, XGBoost, and MDS from the same L3.3 state by
#   reusing the same ``FluxLevelData`` — no upstream re-runs.
# - **Pure functions**: each call returns a new container, never mutates the
#   input — easy to unit-test, easy to reason about.
