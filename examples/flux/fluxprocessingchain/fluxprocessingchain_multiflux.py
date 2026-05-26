"""
=======================================================
Flux Processing Chain - Multi-Flux Loop
=======================================================

``run_flux_chain`` runs the full L2 -> L4.1 pipeline for one flux variable.
``FluxConfig`` captures every per-flux setting (outlier thresholds, USTAR
values, gap-filling features, L2 tests) so that site-level parameters
(coordinates, QCF thresholds) are written once and shared across all fluxes.

Typical site dataset includes:

- **FC / NEE** — CO2 flux with storage correction and USTAR filtering
- **H**        — Sensible heat flux (no USTAR filtering for energy fluxes)
- **LE**       — Latent heat flux  (no USTAR filtering)
- **N2O**      — Nitrous oxide flux (trace gas, lower outlier sigma)
- **CH4**      — Methane flux       (trace gas, lower outlier sigma)

This example processes FC, H, and N2O — the most common combination.

**Before running:**
Determine site-specific USTAR thresholds via a bootstrap analysis in
REddyProc (R package).  Pass the 16th, 50th, and 84th percentile values
as ``ustar_thresholds`` to quantify sensitivity.

**VPD unit warning:**
MDS requires VPD in kPa.  EddyPro outputs VPD in hPa — divide by 10
before assigning to ``mds_vpd`` in ``FluxConfig``.
"""

# %%
# Imports
# ^^^^^^^

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.flux.fluxprocessingchain import FluxConfig, run_flux_chain

# %%
# Load data
# ^^^^^^^^^

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.loc['2024-06':'2024-06']  # one month for speed

# %%
# Site-level parameters (written once, shared across all fluxes)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These describe the measurement site and define what QCF level is
# considered acceptable.  FLUXNET / Swiss FluxNet convention:
#   QCF=0  — all tests pass  (always kept)
#   QCF=1  — soft warnings   (kept by default with accept_qcf_below=2)
#   QCF=2  — hard failure    (always discarded)

SITE = dict(
    site_lat=47.41887,  # CH-HON
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,  # W m-2; SW_IN below this = nighttime
    daytime_accept_qcf_below=2,  # keep QCF=0 and QCF=1 during daytime
    nighttime_accept_qcf_below=2,
)

# %%
# Feature engineer (shared by all ML gap-filling runs)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Build one FeatureEngineer and pass it to every ``run_flux_chain`` call.
# Feature engineering runs once per call — the same instance is reused
# internally for RF and XGBoost so no work is repeated.
#
# All feature columns must exist in the input DataFrame.

FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

engineer = FeatureEngineer(
    target_col='_target_',  # placeholder; not used for L4.1 features
    features_lag=[-2, -1],
    features_lag_stepsize=1,
    features_rolling=[2, 4, 12, 24, 48],
    features_rolling_stats=['median', 'min', 'max', 'std'],
    features_diff=[1, 2],
    features_ema=[6, 12, 24, 48],
    features_poly_degree=2,
    features_stl=True,
    features_stl_method='stl',
    features_stl_seasonal_period=48,
    features_stl_components=['trend', 'seasonal', 'residual'],
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    sanitize_timestamp=True,
    verbose=1,
)

# %%
# Per-flux configurations
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# FC / NEE — CO2 flux
# -------------------
# Storage correction uses the measured single-point storage term SC_SINGLE.
# USTAR filtering removes nighttime low-turbulence records (threshold from
# REddyProc bootstrap; use 16th/50th/84th percentiles in production).
# MDS driver columns use gap-filled meteorology to avoid introducing gaps.
#
# NOTE: mds_vpd must be in kPa.  This example uses a pre-computed kPa column;
#       if your column is in hPa (EddyPro default), divide by 10 first.

fc_cfg = FluxConfig(
    fluxcol='FC',
    ustar_thresholds=[0.30],  # m s-1 — 50th percentile from REddyProc
    ustar_labels=['CUT_50'],
    outlier_sigma_daytime=5.5,  # Hampel sigma for daytime NEE
    outlier_sigma_nighttime=5.5,  # Hampel sigma for nighttime NEE
    gapfilling_features=FEATURES,
    level2_tests={
        'ssitc': {'apply': True, 'setflag_timeperiod': None},
        'gas_completeness': {'apply': True},
        'spectral_correction_factor': {'apply': True},
        'signal_strength': {
            'apply': True,
            'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
            'method': 'discard below',
            'threshold': 60,
        },
        'raw_data_screening_vm97': {
            'apply': True,
            'spikes': True, 'amplitude': False, 'dropout': True,
            'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
            'discont_hf': False, 'discont_sf': False,
        },
    },
    set_storage_to_zero=False,  # storage measurement available
    gapfill_rf=True,
    gapfill_xgb=False,  # skip for speed in this demo
    gapfill_mds=True,
    mds_swin='SW_IN_T1_47_1_gfXG',
    mds_ta='TA_T1_47_1_gfXG',
    mds_vpd='VPD_T1_47_1_gfXG',  # kPa; see unit warning above
)

# %%
# H — Sensible heat flux
# ----------------------
# Energy fluxes are NOT subject to USTAR filtering.  Pass thresholds=[0.0]
# and labels=['CUT_NONE'] — USTAR is always >= 0, so no records are removed.
#
# Storage correction for H uses the single-point approach; if no profile
# measurement is available for your site, set set_storage_to_zero=True.

h_cfg = FluxConfig(
    fluxcol='H',
    ustar_thresholds=[0.0],  # no USTAR filtering for energy fluxes
    ustar_labels=['CUT_NONE'],
    outlier_sigma_daytime=5.5,
    outlier_sigma_nighttime=5.5,
    gapfilling_features=FEATURES,
    level2_tests={
        'ssitc': {'apply': True, 'setflag_timeperiod': None},
        'raw_data_screening_vm97': {
            'apply': True,
            'spikes': True, 'amplitude': False, 'dropout': True,
            'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
            'discont_hf': False, 'discont_sf': False,
        },
    },
    set_storage_to_zero=True,  # no heat storage profile — set correction to zero
    gapfill_rf=True,
    gapfill_xgb=False,
    gapfill_mds=True,
    mds_swin='SW_IN_T1_47_1_gfXG',
    mds_ta='TA_T1_47_1_gfXG',
    mds_vpd='VPD_T1_47_1_gfXG',
)

# %%
# N2O — Nitrous oxide flux (trace gas)
# ------------------------------------
# Trace gas fluxes are noisier than CO2 — the signal-to-noise ratio is lower
# and the distribution has heavier tails.  The Hampel sigma thresholds must
# be chosen by visually inspecting the N2O record; values that work for CO2
# (5–6) will over-smooth N2O.  No universal default applies.
#
# MDS is disabled here because the N2O-specific meteorological drivers that
# would be needed (soil temperature, moisture) are often unavailable.

n2o_cfg = FluxConfig(
    fluxcol='N2O',
    ustar_thresholds=[0.30],
    ustar_labels=['CUT_50'],
    outlier_sigma_daytime=4.0,  # lower sigma needed for trace gases
    outlier_sigma_nighttime=3.5,  # chosen by inspecting the N2O time series
    gapfilling_features=FEATURES,
    level2_tests={
        'ssitc': {'apply': True, 'setflag_timeperiod': None},
        'gas_completeness': {'apply': True},
    },
    set_storage_to_zero=False,
    gapfill_rf=True,
    gapfill_xgb=False,
    gapfill_mds=False,  # MDS not appropriate without dedicated N2O drivers
)

# %%
# Run the full chain for each flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``run_flux_chain`` runs L1 -> L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1 for
# one flux, guided entirely by its ``FluxConfig``.
#
# Results are stored in a dict keyed by the flux column name so that
# downstream code can access each result independently.
#
# The same ``engineer`` instance is passed to every call — feature
# engineering is computed once per call and reused internally.

configs = [fc_cfg, h_cfg]  # skip N2O in this demo (column not in example data)

results: dict[str, dv.flux.FluxLevelData] = {}
for cfg in configs:
    print(f"\n{'=' * 60}")
    print(f"Processing {cfg.fluxcol} ...")
    print('=' * 60)
    results[cfg.fluxcol] = run_flux_chain(
        df, cfg,
        **SITE,
        engineer=engineer,
        showplot=False,
        verbose=True,
        # Demo settings — use production defaults (no overrides) in real runs
        rf_kwargs=dict(n_estimators=2, max_depth=1),
    )

# %%
# Inspect results per flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^

for fluxcol, data in results.items():
    print(f"\n{fluxcol} — data availability summary:")
    print(data.summary())

# %%
# Gap-filled column names and gap-filling fractions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.gapfilled_cols()`` returns a nested dict
# ``{method: {ustar_scenario: column_name}}`` — the gap-filled column in
# ``data.fpc_df`` for each combination of method and USTAR scenario.
#
# The FLAG_*_ISFILLED column encodes the origin of every value:
#   0 = directly measured (passed all QC filters)
#   1 = gap-filled by the primary model
#   2 = fallback (primary model could not predict; very rare)

for fluxcol, data in results.items():
    cols = data.gapfilled_cols()
    print(f"\n{fluxcol} gap-filled columns: {cols}")
    for method_key, scen_cols in cols.items():
        for scen, gf_col in scen_cols.items():
            flag_col = f"FLAG_{gf_col}_ISFILLED"
            if flag_col in data.fpc_df.columns:
                flags = data.fpc_df[flag_col]
                n_measured = int((flags == 0).sum())
                n_filled = int((flags == 1).sum())
                total = n_measured + n_filled
                pct = 100 * n_filled / max(total, 1)
                print(f"  {method_key} {scen}: {n_filled}/{total} gap-filled ({pct:.1f}%)")

# %%
# Export: combine all fluxes into a single DataFrame
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each ``data.fpc_df`` shares the same DatetimeIndex as the input ``df``.
# Concatenate along axis=1 and drop duplicate columns (site flags shared
# across all fluxes, e.g. daytime/nighttime, USTAR).
#
# The combined DataFrame contains:
#   - All raw flag and QCF columns per flux
#   - Storage-corrected flux columns (L3.1)
#   - Gap-filled flux columns + their ISFILLED flags (L4.1)
#
# **Next step — flux partitioning:**
# Gap-filled NEE can be partitioned into gross primary production (GPP) and
# ecosystem respiration (Reco) using the nighttime or daytime method.
# Use an external tool such as REddyProc (R) for this step.

combined = results['FC'].fpc_df.copy()
for fluxcol, data in results.items():
    if fluxcol == 'FC':
        continue
    new_cols = [c for c in data.fpc_df.columns if c not in combined.columns]
    combined = combined.join(data.fpc_df[new_cols], how='left')

print(f"\nCombined export DataFrame: {combined.shape[0]} rows x {combined.shape[1]} cols")

# To save to disk:
# dv.save_parquet(combined, filename='flux_multiflux_L41', outpath='.')
