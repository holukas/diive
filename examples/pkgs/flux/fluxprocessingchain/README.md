# Flux Processing Chain Examples

Multi-level eddy covariance flux processing following Swiss FluxNet standards. Takes raw measurements through quality control, storage correction, outlier detection, USTAR filtering, and gap-filling.

## Contents

- `fluxprocessingchain.py` — Full pipeline demonstration (Levels 2-3.1)

## How It Works

The processing runs sequentially through five levels:

- **Level-2**: Quality control with 7 configurable tests (SSITC, gas completeness, signal strength, VM97 raw data, spectral correction, angle of attack, wind steadiness)
- **Level-3.1**: Add storage correction term to flux, optionally gap-fill missing values
- **Level-3.2**: Remove outliers using sequential detection methods (Hampel, z-score, LOF, etc.)
- **Level-3.3**: USTAR threshold filtering to flag low-turbulence nighttime data
- **Level-4.1**: Gap-fill remaining gaps with Random Forest, XGBoost, or meteorological similarity

## Methods

### Getting Data

- `get_data(verbose=1)` — Full output DataFrame (input + all processing results)
- `get_gapfilled_names()` — Which variables got gap-filled
- `get_nongapfilled_names()` — Which variables didn't  
- `get_gapfilled_variables()` — Just the gap-filled columns

### Level-2: Quality Control

- `level2_quality_flag_expansion(...)` — Run the 7 quality tests
- `finalize_level2()` — Commit results and apply quality threshold

### Level-3.1: Storage Term

- `level31_storage_correction(gapfill_storage_term=True, set_storage_to_zero=False)` — Add storage term, optionally fill gaps with rolling median
- `finalize_level31()` — Commit and apply quality threshold

### Level-3.2: Outlier Detection

- `level32_stepwise_outlier_detection()` — Start the chain
- `level32_flag_outliers_hampel_dtnt_test(...)` — Hampel filter (day/night separate)
- `level32_flag_outliers_zscore_dtnt_test(...)` — Z-score (day/night separate)
- `level32_flag_outliers_zscore_rolling_test(...)` — Rolling z-score with adaptive threshold
- `level32_flag_outliers_zscore_test(...)` — Global z-score
- `level32_flag_outliers_localsd_test(...)` — Local standard deviation
- `level32_flag_outliers_abslim_dtnt_test(...)` — Min/max bounds (day/night separate)
- `level32_flag_outliers_abslim_test(...)` — Min/max bounds (global)
- `level32_flag_outliers_increments_zcore_test(...)` — Detect sudden jumps
- `level32_flag_outliers_lof_test(...)` — Local Outlier Factor
- `level32_flag_outliers_lof_dtnt_test(...)` — LOF (day/night separate)
- `level32_flag_outliers_trim_low_test(...)` — Symmetric trimming
- `level32_flag_manualremoval_test(remove_dates=[], ...)` — Manually remove specific periods
- `level32_addflag()` — Commit current test before running the next one
- `finalize_level32()` — Commit and apply quality threshold

### Level-3.3: USTAR Filtering

- `level33_constant_ustar(thresholds=[...], threshold_labels=[...], ...)` — Apply USTAR thresholds (typically multiple percentiles for uncertainty)
- `finalize_level33()` — Commit and apply quality threshold

### Level-4.1: Gap-Filling

- `level41_mds(swin, ta, vpd, swin_tol, ta_tol, vpd_tol, ...)` — Meteorological similarity (no training)
- `level41_longterm_random_forest(features=[...], features_lag=[...], ...)` — Random Forest (trained, interpretable)
- `level41_longterm_xgboost(features=[...], features_lag=[...], ...)` — XGBoost (trained, fast)

### Analysis & Visualization

- `analyze_highest_quality_flux(showplot=True)` — Filter best records using Hampel spike detection
- `showplot_gapfilled_heatmap(...)` — Time series heatmap
- `showplot_gapfilled_cumulative(...)` — Cumulative sum over time
- `showplot_feature_ranks_per_year()` — Feature importance per year
- `showplot_mds_gapfilling_qualities()` — MDS match quality by date

### Reports

- `report_gapfilling_variables()` — How many records gapfilled per variable
- `report_traintest_model_scores(outpath=None)` — Model accuracy metrics
- `report_traintest_details(outpath=None)` — Training/test split diagnostics
- `report_gapfilling_model_scores(outpath=None)` — Final model performance
- `report_gapfilling_feature_importances(outpath=None)` — Which features matter
- `report_gapfilling_poolyears()` — Multi-year results combined

### Properties (Read-Only)

- `df` — Original input
- `fpc_df` — Working copy with all intermediate results
- `filteredseries` — Final flux after all processing
- `filteredseries_hq` — Only the best records (QCF=0)
- `filteredseries_level2_qcf` through `filteredseries_level33_qcf` — Series at each stage
- `level2`, `level31`, `level32`, `level33`, `level41` — Access detailed results from each level

## Output Variables

The processing adds columns to your DataFrame at each step:

### Level-2: Quality Flags

Each test produces a flag (0=pass, 1=soft warning, 2=fail):
- `FLAG_SSITC_L2_...`, `FLAG_GAS_COMPLETENESS_L2_...`, etc. (one per test)
- `QCF_L2_...` — Combined quality rating (0=good, 1=marginal, 2=poor)

### Level-3.1: Storage Correction

Storage term gets added to flux, gaps filled with rolling median if enabled:
- `{fluxname}_L3.1` — Corrected flux (e.g., `NEE_L3.1`, `LE_L3.1`)
- `{strgcol}_gfRMED_L3.1` — Gap-filled storage (e.g., `SC_SINGLE_gfRMED_L3.1`)
- `FLAG_{strgcol}_gfRMED_L3.1_ISFILLED` — Which storage values were filled (0=original, 1=filled)
- `QCF_L3.1_...` — Quality after storage correction

### Level-3.2: Outliers

Each detection method adds its own flag:
- `FLAG_HAMPEL_L32_...`, `FLAG_ZSCORE_L32_...`, etc.
- `QCF_L3.2_...` — Combined quality

### Level-3.3: USTAR

Separate results for each USTAR scenario:
- `FLAG_USTAR_CUT_50_...`, `FLAG_USTAR_CUT_16_...`, etc. (or whatever thresholds you set)
- `QCF_L3.3_..._CUT_50` (one per scenario)

### Level-4.1: Gap-Filled

- `{fluxname}_L4.1_GAPFILLED_RF_CUT_50` — Gap-filled values using Random Forest for this USTAR scenario
- `{fluxname}_L4.1_ISGAPFILLED_RF_CUT_50` — Which values were filled
- Feature columns (if ML methods): lag features, rolling stats, differences, EMA, polynomial, STL components, timestamp features, record number

### Added at Startup

- `SW_IN_POT` — Potential radiation (calculated from site lat/lon)
- `DAYTIME` — 1 if day, 0 if night (based on solar elevation)
- `NIGHTTIME` — inverse of DAYTIME

## How to Get Results

```python
# Everything together
full_data = fpc.get_data(verbose=1)

# Specific level results
qc_flags = fpc.level2.results
storage_info = fpc.level31.results
outlier_flags = fpc.level32.results
gap_filled = fpc.level41['RF']['CUT_50'].gapfilled_

# Series at different points
fpc.filteredseries                 # Final output
fpc.filteredseries_hq              # Only best quality (QCF=0)
fpc.filteredseries_level2_qcf      # After Level-2
fpc.filteredseries_level31_qcf     # After Level-3.1
```

## Gap-Filling Details

All three gap-filling methods (Random Forest, XGBoost, MDS) use the same 8-stage feature engineering pipeline. The difference is in how the model uses these features.

### Feature Engineering (8 Stages, Shared by All Methods)

1. **Lag features** (`features_lag=[-2, -1]`): Past 30-60 min context
   - Step size 1 means use every timestep, not every 2nd
   
2. **Rolling stats** (`features_rolling=[2, 4, 12, 24, 48]`): Windows of 1hr, 2hr, 6hr, 12hr, 24hr (for 30-min data)
   - Captures patterns repeating daily or at longer timescales
   - Stats: median (robust to outliers), min, max, std, 25th and 75th percentiles
   
3. **Differencing** (`features_diff=[1, 2]`): Rate of change and acceleration
   - 1st order catches transitions, 2nd order catches rapid shifts
   
4. **EMA** (`features_ema=[6, 12, 24, 48]`): Exponential moving averages at 3hr, 6hr, 12hr, 24hr
   - Lagged system responses
   
5. **Polynomial** (`features_poly_degree=2`): Quadratic terms
   - Captures non-linear saturation curves
   
6. **STL** (`features_stl=True`): Separate trend from periodic patterns
   - Daily cycle period = 48 (24 hours at 30-min data)
   - Useful when your signal has clear daily and seasonal variations
   
7. **Timestamps** (`vectorize_timestamps=True`): Year, season, day-of-year, hour, etc.
   - ~19 features total, captures time-of-day and seasonal effects
   - Uses sine/cosine encoding for circular variables (hour, day-of-year, month, day-of-week) to preserve cyclical continuity (e.g., hour 23 is close to hour 0, not far)
   
8. **Record number** (`add_continuous_record_number=True`): Just 1, 2, 3, ... to row number
   - Detects long-term drift in your system

### Random Forest Configuration

Random Forest works by training an ensemble of trees and averaging their predictions. It's interpretable and handles outliers well.

**Key parameters:**

- `n_estimators=350` — Number of trees. RF needs ~50% more than XGBoost to reach similar accuracy. Increase if underfitting (low R²), decrease if overfitting or memory-limited.
- `max_depth=15` — How deep trees can grow. RF tolerates deeper trees than XGBoost without overfitting. Default 15 is a good balance.
- `min_samples_split=5` — Minimum records required to split a node. Prevents fitting to noise.
- `min_samples_leaf=2` — Minimum records at leaf nodes. Higher values = smoother, less overfit predictions.
- `reduce_features=True` — SHAP-based feature selection. Keeps only important features across all years, speeds up training, but may miss year-specific patterns.
- `n_jobs=-1` — Use all CPU cores for parallel training.
- `random_state=42` — Set seed for reproducible results.

### XGBoost Configuration

XGBoost uses sequential boosting where each new tree corrects errors from the previous ones. It's faster and often more accurate on non-linear patterns.

**Key parameters:**

- `n_estimators=500` — Number of boosting rounds. XGBoost needs fewer estimators than Random Forest to reach similar performance. Increase if underfitting, decrease if overfitting.
- `max_depth=6` — Tree depth (shallow, prevents overfitting). Default 6 is good; increase to 7-8 for complex patterns, decrease to 4-5 if overfitting.
- `learning_rate=0.05` — Shrinkage parameter controlling how fast the model learns. Smaller values (0.05) = better generalization but slower training; 0.1 is standard; 0.3 = aggressive.
- `early_stopping_rounds=30` — Stop training if validation doesn't improve for N rounds. Prevents overfitting and reduces unnecessary training.
- `min_child_weight=5` — Minimum sum of weights required in a child node (similar to `min_samples_leaf`).
- `n_jobs=-1` — Use all CPU cores.
- `random_state=42` — Reproducible results.

### MDS (Meteorological Data Similarity)

MDS doesn't require training. Instead, it fills gaps by finding meteorologically similar records and averaging their flux values. No model is learned, so no risk of overfitting.

**Critical requirement:** All three meteorological variables must be specified and their units must be exact.

**Key parameters:**

- `swin` — Solar radiation variable name. **Must be in W/m²** (not µmol).
- `ta` — Temperature variable name. **Must be in °C** (not K).
- `vpd` — Vapor pressure deficit variable name. **Must be in hPa** (not Pa).
- `swin_tol=[20, 50]` — Two-level radiation tolerance (W/m²). Finds records within first band, then relaxes to second band if few matches.
- `ta_tol=2.5` — Temperature tolerance (°C). How close other records must match.
- `vpd_tol=0.5` — VPD tolerance (hPa). How close other records must match.
- `avg_min_n_vals=5` — Minimum matching records needed to average a value.

**Note:** Wrong units produce silent failures. Double-check that radiation is W/m² and not µmol m⁻² s⁻¹, temperature is °C and not K, VPD is hPa and not Pa.

### Method Comparison

| Aspect | Random Forest | XGBoost | MDS |
|--------|---|---|---|
| **Training required** | Yes | Yes | No |
| **Interpretable** | Yes | Limited | Yes (simple averaging) |
| **Typical R²** | 0.60–0.80 | 0.65–0.85 | 0.40–0.70 |
| **Handles outliers** | Better | Good | Good (meteorological filtering) |
| **Non-linear patterns** | Good | Better | No (matching-based) |
| **Model size** | 2–3× larger | Smaller | None (just matching logic) |
| **Overfitting risk** | Moderate | Lower | None |

## Running

```bash
uv run python examples/pkgs/flux/fluxprocessingchain/fluxprocessingchain.py
```

Or all examples:
```bash
uv run python examples/run_all_examples.py
```

## More Info

Core classes are in `diive.pkgs.flux.fluxprocessingchain`:
- `FluxProcessingChain` — Main orchestrator
- `FluxQualityFlagsEddyPro`, `FluxStorageCorrectionSinglePointEddyPro`, `StepwiseOutlierDetection`, `FlagMultipleConstantUstarThresholds` — Individual level handlers

Gap-filling models in `diive.pkgs.gapfilling`:
- `LongTermGapFillingRandomForestTS`, `LongTermGapFillingXGBoostTS`, `FluxMDS`
