"""
FLUX PROCESSING CHAIN: COMPLETE WORKFLOW EXAMPLE
================================================

Swiss FluxNet-compliant post-processing across five levels: quality control, storage correction,
outlier detection, USTAR filtering, and gap-filling. Demonstrates both Random Forest and XGBoost.

Level-3.3 shows two approaches for USTAR filtering:

* **Option A (recommended for new sites / first run):** Auto-detect the USTAR threshold
  from the data using a bootstrap algorithm (``level33_ustar_detection``).
* **Option B (for repeat runs or known sites):** Apply a pre-known constant threshold
  (``level33_constant_ustar``).

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load data and configure site parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start with raw 30-minute flux data and set site-specific parameters for
# the processing chain.  The full available dataset is used here because USTAR
# threshold detection requires at least ~3000 nighttime records to be reliable.

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN

df = load_exampledata_parquet_lae_level1_30MIN()

# Site and processing parameters
FLUXVAR = "FC"
SITE_LAT = 47.41887  # CH-HON
SITE_LON = 8.491318  # CH-HON
UTC_OFFSET = 1
NIGHTTIME_THRESHOLD = 20  # W m-2, conditions below are nighttime
DAYTIME_ACCEPT_QCF_BELOW = 2
NIGHTTIME_ACCEPT_QCF_BELOW = 2

# Initialize processing chain
fpc = dv.flux.FluxProcessingChain(
    df=df,
    fluxcol=FLUXVAR,
    site_lat=SITE_LAT,
    site_lon=SITE_LON,
    utc_offset=UTC_OFFSET,
    nighttime_threshold=NIGHTTIME_THRESHOLD,
    daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
    nighttime_accept_qcf_below=NIGHTTIME_ACCEPT_QCF_BELOW
)

# %%
# Level-2: Quality Control Tests
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Apply 7 configurable quality tests following EddyPro standards:
# - SSITC (steady-state and integral turbulence characteristics)
# - Gas completeness (availability of required variables)
# - Signal strength (IRGA diagnostic quality)
# - Spectral correction factor
# - VM97 raw data tests (spikes, amplitude, dropout, etc.)
# - Angle of attack
# - Steadiness of horizontal wind

TEST_SSITC = True
TEST_SSITC_SETFLAG_TIMEPERIOD = None
TEST_GAS_COMPLETENESS = True
TEST_SPECTRAL_CORRECTION_FACTOR = True
TEST_SIGNAL_STRENGTH = True
TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN'
TEST_SIGNAL_STRENGTH_METHOD = 'discard below'
TEST_SIGNAL_STRENGTH_THRESHOLD = 60
TEST_RAWDATA = True
TEST_RAWDATA_SPIKES = True
TEST_RAWDATA_AMPLITUDE = False
TEST_RAWDATA_DROPOUT = True
TEST_RAWDATA_ABSLIM = False
TEST_RAWDATA_SKEWKURT_HF = False
TEST_RAWDATA_SKEWKURT_SF = False
TEST_RAWDATA_DISCONT_HF = False
TEST_RAWDATA_DISCONT_SF = False
TEST_RAWDATA_ANGLE_OF_ATTACK = False
TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False
TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND = False

LEVEL2_SETTINGS = {
    'signal_strength': {
        'apply': TEST_SIGNAL_STRENGTH,
        'signal_strength_col': TEST_SIGNAL_STRENGTH_COL,
        'method': TEST_SIGNAL_STRENGTH_METHOD,
        'threshold': TEST_SIGNAL_STRENGTH_THRESHOLD},
    'raw_data_screening_vm97': {
        'apply': TEST_RAWDATA,
        'spikes': TEST_RAWDATA_SPIKES,
        'amplitude': TEST_RAWDATA_AMPLITUDE,
        'dropout': TEST_RAWDATA_DROPOUT,
        'abslim': TEST_RAWDATA_ABSLIM,
        'skewkurt_hf': TEST_RAWDATA_SKEWKURT_HF,
        'skewkurt_sf': TEST_RAWDATA_SKEWKURT_SF,
        'discont_hf': TEST_RAWDATA_DISCONT_HF,
        'discont_sf': TEST_RAWDATA_DISCONT_SF},
    'ssitc': {
        'apply': TEST_SSITC,
        'setflag_timeperiod': TEST_SSITC_SETFLAG_TIMEPERIOD},
    'gas_completeness': {
        'apply': TEST_GAS_COMPLETENESS},
    'spectral_correction_factor': {
        'apply': TEST_SPECTRAL_CORRECTION_FACTOR},
    'angle_of_attack': {
        'apply': TEST_RAWDATA_ANGLE_OF_ATTACK,
        'application_dates': TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES},
    'steadiness_of_horizontal_wind': {
        'apply': TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND}
}

fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
fpc.finalize_level2()

# %%
# Level-3.1: Storage Correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add the single-point profile gas storage term to the flux measurement.
# Gap-fill missing storage values using expanding-window rolling median.
# This correction typically adds 2-3% to the measured flux.

fpc.level31_storage_correction(gapfill_storage_term=True, set_storage_to_zero=False)
fpc.finalize_level31()

# %%
# Level-3.2: Outlier Detection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Remove outliers using a sequential chain of detection methods.
# Starting with Hampel filter (robust spike detection), then additional
# tests can be chained. Each test operates on data already filtered by
# previous tests. Preview plots help verify each stage before committing.

fpc.level32_stepwise_outlier_detection()
fpc.level32_flag_outliers_hampel_test(
    window_length=48 * 13,
    n_sigma_daytime=5.5,
    n_sigma_nighttime=5.5,
    showplot=False,
    verbose=True,
    use_differencing=True,
    separate_daytime_nighttime=True,
    repeat=True
)
fpc.level32_addflag()
fpc.finalize_level32()

# Show quality control heatmap
fpc.level32_qcf.showplot_qcf_heatmaps()

# %%
# Level-3.3: USTAR Filtering -- Option A: Auto-detect threshold (recommended)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Remove low-turbulence nighttime periods using friction velocity (USTAR)
# threshold filtering.  USTAR filtering applies only to CO2/CH4/N2O fluxes
# (not energy fluxes H or LE).
#
# Option A runs a bootstrap algorithm (ONEFlux moving point, Papale et al. 2006)
# on the storage-corrected flux to find the threshold automatically.  The result
# is a CUT (constant upper threshold) derived from the pooled bootstrap distribution.
#
# ``percentiles=(50,)`` produces a single filtering scenario labelled ``CUT_50``.
# For full uncertainty quantification pass ``percentiles=(16, 50, 84)`` to get
# three scenarios (``CUT_16``, ``CUT_50``, ``CUT_84``).
#
# ``n_iter=10`` here for demo speed -- use 100 or more in production.
# ``n_jobs=-1`` runs bootstrap iterations in parallel across all available CPUs.

TA_COL = 'TA_T1_47_1_gfXG'      # Air temperature column in the input data (deg C)
SWIN_COL = 'SW_IN_T1_47_1_gfXG'  # Incoming shortwave radiation column (W m-2)

fpc.level33_ustar_detection(
    ta_col=TA_COL,
    swin_col=SWIN_COL,
    n_iter=10,  # Demo setting (use 100+ in production)
    n_jobs=-1,
    percentiles=(50,),  # Single scenario CUT_50; use (16, 50, 84) for uncertainty range
    showplot=False,
    verbose=True,
)

# Inspect detection results
print(fpc.ustar_detection.summary())

# The detected CUT threshold (p50) that was applied
cut_threshold = fpc.ustar_detection.get_cut_threshold()['p50']
print(f"\nDetected USTAR threshold (CUT p50): {cut_threshold:.4f} m/s")
print("Use this value in subsequent runs with level33_constant_ustar() for speed.")

# %%
# Level-3.3: USTAR Filtering -- Option B: Pre-known constant threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Once you have a trusted site-specific threshold (e.g. from a previous
# detection run or from REddyProc), skip detection and apply the value directly.
# This is faster and fully reproducible across processing runs.
#
# To switch to this option, comment out ``level33_ustar_detection`` above and
# uncomment the block below.  Keep the scenario label consistent with what
# gap-filling expects downstream (e.g. ``'CUT_50'``).

# fpc.level33_constant_ustar(
#     thresholds=[0.30],       # Site-specific threshold from prior detection (m/s)
#     threshold_labels=['CUT_50'],
#     showplot=False,
# )

# %%
# Level-4.1: Gap-Filling with Machine Learning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Fill remaining gaps using trained ML models or meteorological similarity.
# Both Random Forest and XGBoost use the same 8-stage feature engineering
# pipeline. We'll run both for comparison.
#
# Features for gap-filling:

FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

# Shared gap-filling parameters (RF and XGB)
GAPFILLING_PARAMS = {
    'features': FEATURES,
    # Feature Engineering (8-stage pipeline)
    'features_lag': [-2, -1],
    'features_lag_stepsize': 1,
    'features_lag_exclude_cols': None,
    'features_rolling': [2, 4, 12, 24, 48],
    'features_rolling_exclude_cols': None,
    'features_rolling_stats': ['median', 'min', 'max', 'std', 'q25', 'q75'],
    'features_diff': [1, 2],
    'features_diff_exclude_cols': None,
    'features_ema': [6, 12, 24, 48],
    'features_ema_exclude_cols': None,
    'features_poly_degree': 2,
    'features_poly_exclude_cols': None,
    'features_stl': True,
    'features_stl_method': 'stl',
    'features_stl_seasonal_period': 48,
    'features_stl_exclude_cols': None,
    'features_stl_components': ['trend', 'seasonal', 'residual'],
    'vectorize_timestamps': True,
    'add_continuous_record_number': True,
    'sanitize_timestamp': True,
    'reduce_features': True,
    'verbose': True,
    'n_jobs': -1,
    'random_state': 42,
}

# %%
# Random Forest Gap-Filling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Train an ensemble of decision trees. Interpretable results, robust to outliers.

fpc.level41_longterm_random_forest(
    **GAPFILLING_PARAMS,
    # RF-specific hyperparameters
    n_estimators=2,  # Demo setting (use 350+ in production)
    max_depth=1,  # Demo setting (use 15 in production)
    min_samples_split=5,
    min_samples_leaf=2,
)

print("Random Forest gap-filling complete")

# %%
# XGBoost Gap-Filling
# ^^^^^^^^^^^^^^^^^^^
#
# Train gradient boosted trees. Better non-linear patterns, faster training.

fpc.level41_longterm_xgboost(
    **GAPFILLING_PARAMS,
    # XGB-specific hyperparameters
    n_estimators=2,  # Demo setting (use 350+ in production)
    max_depth=1,  # Demo setting (use 6-8 in production)
    learning_rate=0.05,
    early_stopping_rounds=30,
    min_child_weight=5,
)

print("XGBoost gap-filling complete")

# %%
# Extract and display results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Access final gap-filled data and model performance metrics.

results = fpc.get_data()
gapfilled_names = fpc.get_gapfilled_names()
print(f"Variables gap-filled: {gapfilled_names}")

# Report on gap-filling success
fpc.report_gapfilling_variables()
fpc.report_gapfilling_model_scores()
fpc.report_traintest_model_scores()

# %%
# Access model-specific results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Retrieve performance metrics and gap-filled data for each method.

rf_model = fpc.level41['long_term_random_forest']['CUT_50']
xgb_model = fpc.level41['long_term_xgboost']['CUT_50']

# Get R² scores (scores_ is a dict with year keys)
rf_r2 = list(rf_model.scores_.values())[0]['r2'] if rf_model.scores_ else None
xgb_r2 = list(xgb_model.scores_.values())[0]['r2'] if xgb_model.scores_ else None

if rf_r2 is not None:
    print(f"Random Forest R²: {rf_r2:.3f}")
if xgb_r2 is not None:
    print(f"XGBoost R²: {xgb_r2:.3f}")

# Access final filtered series with all processing applied
# After L3.3 there is no single filtered series; access per USTAR scenario explicitly.
final_flux = fpc.filteredseries_level33_qcf['CUT_50']
highest_quality = fpc.levels.filteredseries_level33_hq['CUT_50']  # Only QCF=0 (best quality)
