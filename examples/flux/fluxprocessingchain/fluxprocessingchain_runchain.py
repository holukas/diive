"""
===============================================
Flux Processing Chain - Single-Call ``run_chain``
===============================================

The flux processing chain ships **two entry points**:

- ``run_chain(data, FluxConfig)`` — single-call driver for the standard
  FLUXNET-style workflow. Fixed sensible defaults for per-detector and
  per-model knobs, only the high-level decisions are on ``FluxConfig``.
- The **composable per-level API** — full control over every knob, every
  detector class, every model hyperparameter. See
  ``fluxprocessingchain_composable.py`` for that path.

This example shows the **simple path**: load data, build one ``FluxConfig``,
hand it to ``run_chain``, inspect the results. Use this when you want the
chain to "just work" and plan to tune later.

The chain that ``run_chain`` runs:

- **L2**   — Quality-flag expansion from EddyPro diagnostics (whatever tests
             you enabled in ``FluxConfig.level2_test_settings``).
- **L3.1** — Single-point storage correction (``FC -> NEE``).
- **L3.2** — Hampel outlier detection with separate day / night sigmas.
             Mandatory because L3.3 depends on outlier-screened data.
- **L3.3** — USTAR filtering, either with a constant threshold you supply
             (``'constant'``) or detected from the data via the FLUXNET
             bootstrap method (``'bootstrap'``).
- **L4.1** — Gap-filling with whichever of MDS / Random Forest / XGBoost
             you enabled. The default ML ``FeatureEngineer`` ships with a
             symmetric ``[-2, 2]`` lag window, first/second-order
             differencing, ``4 / 12 / 48``-record rolling median + std,
             and vectorised timestamps. SHAP feature reduction is **on**.
"""

# %%
# Imports
# ^^^^^^^

from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.flux.fluxprocessingchain import (
    FluxConfig,
    add_driver,
    init_flux_data,
    run_chain,
)

# %%
# Load data
# ^^^^^^^^^
#
# One month from the CH-LAE 2024 example dataset keeps the runtime small.
# The example data already contains ``SW_IN_POT`` / ``DAYTIME`` / ``NIGHTTIME``
# columns from previous processing; we drop them so ``init_flux_data`` can
# compute them fresh (it raises on a collision with reserved names).

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.loc['2024-06':'2024-06']
df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                      if c in df.columns])

# %%
# Step 1 — initialise the container
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``init_flux_data`` calculates potential radiation, derives day / night
# flags, and assembles the frozen ``FluxMeta`` record that every level
# reads from. ``daytime_accept_qcf_below=2`` keeps QCF=0 (all tests
# pass) and QCF=1 (soft warnings); set to ``1`` for the stricter
# QCF=0-only acceptance.

data = init_flux_data(
    df=df,
    fluxcol="FC",
    site_lat=47.41887,  # CH-LAE
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)
print(data)

# %%
# Step 2 — register VPD in kPa
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MDS needs VPD in **kPa**, but EddyPro outputs it in hPa. Convert and
# register the result via ``add_driver`` — this puts the column into
# ``data.full_df`` (where L4.1 reads from), not into ``data.fpc_df``.
# If the column is already in the right unit elsewhere in ``df`` you can
# point ``mds_vpd=`` at it directly and skip this step.

if 'VPD_T1_47_1_gfXG' in df.columns:
    vpd_kpa = (df['VPD_T1_47_1_gfXG'] / 10.0).rename('VPD_kPa')
    data = add_driver(data, vpd_kpa)

# %%
# Step 3 — build the FluxConfig
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Only ``fluxcol`` and ``ustar_thresholds`` are unconditionally required.
# Every other field is required *contextually* by ``run_chain`` based on
# what's enabled: ``outlier_sigma_*`` are needed because L3.2 always runs;
# ``mds_*`` columns are needed because we enabled MDS; ``gapfilling_features``
# is needed because we enabled at least one ML method. If you forget any of
# them, ``run_chain`` raises one cumulative error listing every missing
# field instead of failing partway through.
#
# Omitting ``level2_test_settings`` lets the chain apply
# :data:`DEFAULT_LEVEL2_TEST_SETTINGS` — ``ssitc`` + ``gas_completeness`` +
# ``spectral_correction_factor`` + ``raw_data_screening_vm97`` with
# spikes and dropout enabled. To opt into the analyzer-specific
# signal-strength test, set ``signal_strength_col=...`` on the config.
# To take full control, pass an explicit ``level2_test_settings`` dict.

cfg = FluxConfig(
    fluxcol='FC',
    # USTAR filtering: one constant threshold (e.g. from REddyProc). With
    # multiple thresholds you would also pass ustar_labels=['CUT_16', ...].
    ustar_thresholds=[0.18],
    ustar_labels=['CUT_50'],
    # L3.2 Hampel: chain uses the Hampel filter's own defaults
    # (window_length=48*13 records = 13 days at 30-min sampling,
    # n_sigma_daytime=n_sigma_nighttime=5.5, use_differencing=True,
    # separate day/night). Set outlier_sigma_daytime / outlier_sigma_nighttime
    # / outlier_window_length only when you need to deviate.
    #
    # L2 quality tests: defaults apply (ssitc + gas_completeness + scf +
    # vm97). Uncomment to add the optional signal-strength test:
    signal_strength_col='CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
    # L4.1 MDS (driver columns must already be in data.full_df)
    mds_swin='SW_IN_T1_47_1_gfXG',
    mds_ta='TA_T1_47_1_gfXG',
    mds_vpd='VPD_kPa',
    # L4.1 ML — turn off XGBoost for a faster example; RF stays on.
    gapfill_mds=True,
    gapfill_rf=False,
    gapfill_xgb=True,
    gapfill_reduce_features=True,  # SHAP reduction — the default
    gapfilling_features=['TA_T1_47_1_gfXG', 'SW_IN_T1_47_1_gfXG', 'VPD_kPa'],
)

# %%
# Step 4 — run the chain
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# One call drives every level. Each per-level callable underneath stays
# exactly the same as what the composable example invokes by hand;
# ``run_chain`` just routes the FluxConfig fields to them.

data = run_chain(data, cfg)
print(f"\nlevels run: {data.level_ids}")
print(data.summary())

# %%
# Step 5 — gap-filled columns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``gapfilled_cols()`` returns ``{method: {ustar_scenario: column_name}}``.
# This is the canonical way to discover which columns in ``fpc_df`` hold
# the gap-filled flux — no need to dig through model instances.

cols = data.gapfilled_cols()
print(f"\nGap-filled columns: {cols}")

# %%
# Step 6 — side-by-side comparison plots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``plot_gapfilled_heatmaps`` shows the measured series plus one heatmap
# panel per gap-filling method, all on a shared colour scale.
# ``plot_cumulative_comparison`` overlays the cumulative sums on a single
# axis so you can see how much the methods diverge over the period.
#
# Both helpers default to ``showplot=True``; pass ``showplot=False`` for
# headless / batch use.

data.plot_gapfilled_heatmaps(ustar_scenario='CUT_50', showplot=False)
data.plot_cumulative_comparison(
    ustar_scenario='CUT_50',
    conv_factor=12.011 * 1e-6 * 1800,  # umol CO2 m-2 s-1 -> gC m-2 (per 30-min record)
    units='gC m-2',
    showplot=False,
)

# %%
# Bootstrap USTAR — drop-in alternative
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Replace ``ustar_thresholds``/``ustar_labels`` with the bootstrap-mode
# fields to have the chain detect USTAR thresholds from the data using
# the FLUXNET-standard multi-year bootstrap (ONEFlux moving-point
# method). Three scenarios (``CUT_16`` / ``CUT_50`` / ``CUT_84``) are
# produced by default; the fitted detector lands on
# ``data.levels.ustar_detection`` for post-hoc inspection. ``run_chain``
# applies the constant (CUT) threshold; for per-year VUT filtering use the
# composable ``run_level33_ustar_detection(..., mode='vut')``.
#
# Not run here to keep the example fast::
#
#     cfg_boot = FluxConfig(
#         fluxcol='FC',
#         ustar_detection_mode='bootstrap',
#         ustar_bootstrap_ta_col='TA_T1_47_1_gfXG',
#         ustar_bootstrap_swin_col='SW_IN_T1_47_1_gfXG',
#         ustar_bootstrap_n_iter=100,
#         ustar_bootstrap_percentiles=(16, 50, 84),
#         outlier_sigma_daytime=5.5,
#         outlier_sigma_nighttime=5.5,
#         level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
#         mds_swin='SW_IN_T1_47_1_gfXG',
#         mds_ta='TA_T1_47_1_gfXG',
#         mds_vpd='VPD_kPa',
#         gapfilling_features=['TA_T1_47_1_gfXG', 'SW_IN_T1_47_1_gfXG', 'VPD_kPa'],
#     )
#     data_boot = run_chain(init_flux_data(df, ...), cfg_boot)
#     print(data_boot.levels.ustar_detection.summary())

# %%
# When to drop down to the composable API
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``run_chain`` is intentionally simple — it ships fixed defaults for every
# per-detector / per-model knob. Reach for the composable per-level
# callables (see ``fluxprocessingchain_composable.py``) when you need:
#
# - a non-Hampel L3.2 detector (z-score rolling, abslim, manual removal),
#   or a multi-step L3.2 pipeline
# - L3.3 diagnostic plotting (``showplot=True``) or a custom bootstrap
#   detector class
# - L4.1 MDS tolerances (``swin_tol`` / ``ta_tol`` / ``vpd_tol``)
# - RF / XGBoost hyperparameters (``n_estimators``, ``max_depth``, ...)
# - a custom ``FeatureEngineer`` — use ``make_level41_engineer`` for a
#   default that you can override piecewise
