"""
====================================================
Flux Processing Chain - Composable Functions
====================================================

The flux processing chain is also exposed as standalone pure functions, one per
level.  Each function takes a ``FluxLevelData`` container and returns a new
one — no shared state, no orchestrator class required.

This example runs the **full L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1** pipeline and
demonstrates the key advantage of the composable functional API: **branching** —
running three gap-filling methods (Random Forest, XGBoost, MDS) from the same
L3.3 state without repeating any upstream work.

Pipeline at a glance:
  L2   -> ``run_level2``                — EddyPro quality-flag expansion + QCF
  L3.1 -> ``run_level31``               — storage correction (``FC -> NEE``)
  L3.2 -> ``make_level32_detector`` +
          ``run_level32``               — outlier detection
  L3.3 -> ``run_level33_constant_ustar``— USTAR filtering (constant threshold)
       or ``run_level33_ustar_detection``  (in-pipeline bootstrap detection)
  L4.1 -> ``run_level41_mds`` /
          ``run_level41_rf`` /
          ``run_level41_xgb``           — gap-filling
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
df = df.loc['2024-07':'2024-07']
# The example dataset already contains ``SW_IN_POT`` / ``DAYTIME`` /
# ``NIGHTTIME`` from earlier processing. ``init_flux_data`` reserves these
# names (it computes its own from potential radiation) and raises if any
# is pre-existing. Drop them so the chain can populate fresh values.
df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                      if c in df.columns])

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
# Each test is enabled by passing a config dict with ``{'apply': True, ...}``.
# Omit or pass ``None`` to skip. Recognised test keys: ``ssitc``,
# ``gas_completeness``, ``spectral_correction_factor``, ``signal_strength``,
# ``raw_data_screening_vm97``, ``angle_of_attack``,
# ``steadiness_of_horizontal_wind``. See :func:`run_level2` for per-test
# settings.

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

data.levels.level2_qcf.showplot_qcf_heatmaps()

# %%
# Step 3: Level-3.1 — storage correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Adds the storage term to the flux (``NEE = FC + SC_SINGLE``).
# ``gapfill_storage_term=True`` fills missing storage with a rolling median so
# storage gaps don't propagate. For H / LE without a storage profile, pass
# ``set_storage_to_zero=True`` to keep the chain's L3.1 ordering happy.

data = run_level31(data, gapfill_storage_term=True, set_storage_to_zero=False)
print(f"After L3.1: {data.filteredseries.dropna().count()} valid records "
      f"(col: {data.filteredseries.name})")

# %%
# Step 4: Level-3.2 — outlier removal
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``make_level32_detector`` returns ``(data, sod)`` wired to the right input
# column, ``idstr='L3.2'``, and the site's lat/lon/UTC offset. Add as many
# ``flag_outliers_*`` + ``sod.addflag()`` pairs as you want, then hand the
# detector to ``run_level32``.

data, sod = make_level32_detector(data)
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
# Pass one threshold per scenario; ``threshold_labels`` becomes the key under
# which the result is stored. For multi-scenario sensitivity analysis pass
# e.g. ``thresholds=[0.10, 0.18, 0.25]`` with
# ``threshold_labels=['CUT_16', 'CUT_50', 'CUT_84']``. For energy fluxes
# (H, LE) use ``thresholds=[0], threshold_labels=['CUT_NONE']``. After this
# call ``data.filteredseries`` is ``None`` — access per-scenario results
# via ``data.levels.filteredseries_level33_qcf[<scen>]``.

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
# Alternative: detect USTAR thresholds in-pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``run_level33_ustar_detection`` runs the bootstrap detector and applies the
# requested percentiles as USTAR scenarios in one call. Not run here to keep
# the example fast::
#
#     from diive.flux.fluxprocessingchain import run_level33_ustar_detection
#     data = run_level33_ustar_detection(
#         data,
#         ta_col='TA_1_1_1',
#         swin_col='SW_IN_1_1_1',
#         n_iter=100,
#         percentiles=(16, 50, 84),   # produces CUT_16 / CUT_50 / CUT_84
#         showplot=False, verbose=True,
#     )
#     print(data.levels.ustar_detection.summary())
#
# Same dispatch in ``run_chain`` via
# ``FluxConfig(ustar_detection_mode='bootstrap', ustar_bootstrap_ta_col=...,
# ustar_bootstrap_swin_col=...)``.

# %%
# Step 5b: gap analysis — structure of missing data before gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.gap_stats(level)`` accepts ``'L2'`` / ``'L3.1'`` / ``'L3.2'`` /
# ``'L3.3'`` and returns ``{label: GapStats}``. Call it at any point in the
# chain.

for scen, gs in data.gap_stats('L3.3').items():
    gs.report()  # Rich console report
    gs.showfig(title=f"Gap analysis before gap-filling  --  {scen}")

# %%
# Step 6: feature engineering for ML gap-filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Build a ``FeatureEngineer`` and pass it to both ``run_level41_rf`` and
# ``run_level41_xgb`` — feature engineering runs once and is reused across
# all USTAR scenarios and methods. All feature columns must exist in
# ``data.full_df``. ``target_col='_target_'`` is a placeholder (any string
# not in your feature list works) since L4.1 applies the engineer to
# predictors only.
#
# For sensible 30-min defaults you can override piecewise, use
# :func:`~diive.flux.fluxprocessingchain.make_level41_engineer` instead.

FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

engineer = FeatureEngineer(
    target_col='_target_',  # placeholder; irrelevant for L4.1
    features_lag=[-2, 2],
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
# ``reduce_features=True`` runs SHAP-based feature selection. Demo settings
# below; for production use ``n_estimators >= 350`` and ``max_depth >= 15``.

data = run_level41_rf(
    data,
    features=FEATURES,
    engineer=engineer,  # same instance reused by XGBoost below
    reduce_features=True,  # SHAP-based feature selection
    verbose=1,
    # Demo settings — use n_estimators=350, max_depth=15 in production
    n_estimators=9,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
print("Random Forest gap-filling complete")

# %%
# Step 7b: gap-fill with XGBoost (branch from same L3.3 state)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Reusing the same ``engineer`` and same ``data`` runs L4.1 from the same
# L3.3 state — no upstream re-runs. That's the composable API's main lever.

data = run_level41_xgb(
    data,
    features=FEATURES,
    engineer=engineer,
    reduce_features=True,
    verbose=1,
    # Demo settings — use n_estimators=350, max_depth=6 in production
    n_estimators=99,
    max_depth=6,
    learning_rate=0.05,
    early_stopping_rounds=10,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
)
print("XGBoost gap-filling complete")

# %%
# Step 7c: gap-fill with MDS (branch from same L3.3 state)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Driver columns must exist in ``data.full_df``. MDS requires VPD in
# **kPa** (EddyPro outputs hPa — divide by 10 if needed).
# ``run_level41_mds`` warns automatically when the median VPD looks like
# hPa (> 10) or TA looks like Kelvin.

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
# ``FLAG_*_ISFILLED`` per record: 0 = measured, 1 = primary fill,
# 2 = fallback fill.

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
# Trained models live under ``data.levels.level41_<method>[<scenario>]``.
# ML methods expose ``showplot_feature_ranks_per_year()`` and
# ``scores_`` (per-year R² etc.); MDS exposes ``showplot()``.

rf_model = data.levels.level41_rf['CUT_50']
xgb_model = data.levels.level41_xgb['CUT_50']
mds_model = data.levels.level41_mds['CUT_50']

rf_r2 = list(rf_model.scores_.values())[0]['r2'] if rf_model.scores_ else None
xgb_r2 = list(xgb_model.scores_.values())[0]['r2'] if xgb_model.scores_ else None

# if rf_r2 is not None:
#     print(f"Random Forest  R2: {rf_r2:.3f}")
if xgb_r2 is not None:
    print(f"XGBoost        R2: {xgb_r2:.3f}")

rf_model.showplot_feature_ranks_per_year(title="RF feature importance — NEE CUT_50")
xgb_model.showplot_feature_ranks_per_year(title="XGBoost feature importance — NEE CUT_50")
mds_model.showplot()

# %%
# Step 12: heatmap of gap-filled flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.plot_gapfilled_heatmaps()`` — one panel per method that has been
# run, plus the measured-only series. Shared colour scale.

data.plot_gapfilled_heatmaps(
    ustar_scenario='CUT_50',
    units='umol m-2 s-1',
)

# %%
# Step 13: cumulative gap-filled NEE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Unit conversion µmol CO2 m-2 s-1 -> gC m-2 for 30-min records:
#   12.011 g mol-1  x  1e-6 mol umol-1  x  1800 s per timestep

UMOL_TO_GC = 12.011 * 1e-6 * 1800

# ``data.plot_cumulative_comparison()`` overlays every method that has been
# run on one axes for the chosen USTAR scenario. The dashed grey line shows
# the measured-only series (gaps contribute zero).

data.plot_cumulative_comparison(
    ustar_scenario='CUT_50',
    conv_factor=UMOL_TO_GC,
    units='gC m-2',
)

# %%
# Step 14: export the final dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.fpc_df`` holds every flag, QCF, storage-correction, and gap-filled
# column accumulated by the chain. Use ``data.gapfilled_cols()`` to find the
# gap-filled column for a given (method, scenario); their ISFILLED flag
# columns are ``FLAG_<gf_col>_ISFILLED``.

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
