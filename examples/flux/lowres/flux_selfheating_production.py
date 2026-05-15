"""
=====================================================
Self-Heating Correction: Complete Production Workflow
=====================================================

Complete production workflow for self-heating correction: create scaling factors
table from parallel measurements, then apply to long-term flux data.

This comprehensive example demonstrates the full SCOP methodology:
1. **Calibration phase**: Create reusable scaling factors lookup table from parallel IRGA measurements
2. **Application phase**: Apply the table to long-term data without parallel measurements

Includes multiple visualizations throughout the workflow:
- **Diurnal cycle plots**: Mean daily cycles of correction term (instrument heating patterns)
- **Optimization results**: Scaling factors by USTAR class with uncertainty bounds
- **Correction dashboards**: Before/after flux comparison, correction magnitude, budget impact

Best for: Understanding the complete self-heating correction pipeline from calibration
to operational correction of multi-year flux datasets, with full diagnostic visualizations.
"""

# %%
# Complete SCOP Workflow Overview
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sun-induced heating of open-path IRGA sensors creates spurious negative CO2 flux
# (artificial uptake bias). The SCOP (Self-heating Correction for Open-Path) method
# corrects this in two phases:
#
# **Phase 1: Calibration (Parallel Measurements Period)**
# 1. ScopPhysics: Calculate unscaled flux correction term (FCT_UNSC)
# 2. ScopOptimizer: Optimize scaling factors using closed-path reference
# 3. Output: **Scaling factors lookup table** (20 USTAR classes × day/night)
#
# **Phase 2: Application (Long-term Operational Period)**
# 1. ScopPhysics: Calculate correction term for long-term data
# 2. ScopApplicator: Apply lookup table to correct open-path fluxes
# 3. Output: Corrected flux time series ready for gap-filling and analysis
#
# The scaling factors table remains valid across seasons/years as long as
# instrument configuration is unchanged.

import pandas as pd
import time
from datetime import datetime
from diive.pkgs.flux.lowres.selfheating import ScopPhysics, ScopOptimizer, ScopApplicator
from diive.configs.exampledata import load_exampledata_parquet_lae

print("=" * 80)
print("SCOP Self-Heating Correction: Complete Production Workflow")
print("=" * 80)

# %%
# PHASE 1: CALIBRATION — Create Scaling Factors Table
# ====================================================
#
# Use a period with parallel measurements (open-path IRGA75 + closed-path IRGA72)
# to create the scaling factors lookup table.

print("\n" + "=" * 80)
print("PHASE 1: CALIBRATION — Create Scaling Factors Lookup Table")
print("=" * 80)

# Load parallel measurement data
df_calibration = load_exampledata_parquet_lae()

# Restrict to parallel measurement period
df_calibration = df_calibration.loc["2016-05-27 00:15:00":"2017-12-11 23:45:00"].copy()

print(f"\nCalibration data period: {df_calibration.index.min()} to {df_calibration.index.max()}")
print(f"Total records: {len(df_calibration):,}")
print(f"\nAvailable parallel measurements:")
print(f"  IRGA75 (open-path):  NEE_L3.1_L3.2_QCF_IRGA75, CO2_MOLAR_DENSITY_IRGA75")
print(f"  IRGA72 (closed-path): NEE_L3.1_L3.2_QCF_IRGA72, CO2_MOLAR_DENSITY_IRGA72")

# %%
# Step 1a: Physics-based correction term (Calibration Period)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ScopPhysics calculates the unscaled flux correction term (FCT_UNSC) by modeling
# thermal exchange between instrument surfaces and passing air.

print(f"\n" + "=" * 80)
print("Step 1a: Physics Calculation (Calibration Period)")
print("=" * 80)

tic = time.time()

physics_calib = ScopPhysics(
    flux_type="CO2",
    ta=df_calibration["TA_T1_47_1_gfXG_IRGA72"].copy(),
    gas_density=df_calibration["CO2_MOLAR_DENSITY_IRGA75"].copy() * 1000,
    rho_a=df_calibration["AIR_DENSITY_IRGA72"].copy(),
    rho_v=df_calibration["VAPOR_DENSITY_IRGA72"].copy(),
    u=df_calibration["U_IRGA72"].copy(),
    c_p=df_calibration["AIR_CP_IRGA72"].copy(),
    ustar=df_calibration["USTAR_IRGA72"].copy(),
    lat=47.478333,  # CH–LAE
    lon=8.364389,
    utc_offset=1,
)

physics_calib.run(correction_method_base="JAR09", gapfill=True)
results_physics_calib = physics_calib.get_results()
elapsed = time.time() - tic

print(f"\n[OK] Physics calculation completed in {elapsed:.1f}s")
physics_calib.stats()

# Plot diurnal cycles of correction term
print(f"\nGenerating diurnal cycle plots...")
physics_calib.plot_diel_cycles()

# %%
# Step 2a: Optimize scaling factors (Calibration Period)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ScopOptimizer determines optimal scaling factors in 20 USTAR classes
# (production settings) by comparing open-path flux against closed-path reference.
# Bootstrap runs provide uncertainty quantification.

print(f"\n" + "=" * 80)
print("Step 2a: Scaling Factor Optimization (Calibration Period)")
print("=" * 80)
print("Creating scaling factors table from parallel measurements...")

tic = time.time()

optimizer = ScopOptimizer(
    flux_type="CO2",
    fct_unsc=results_physics_calib["FCT_UNSC_gfRF"],
    class_var=df_calibration["USTAR_IRGA72"].copy(),
    n_classes=20,                                         # Production: 20 classes
    n_bootstrap_runs=100,                                 # Production: 100 runs
    flux_openpath=df_calibration["NEE_L3.1_L3.2_QCF_IRGA75"].copy(),
    flux_closedpath=df_calibration["NEE_L3.1_L3.2_QCF_IRGA72"].copy(),
    daytime=results_physics_calib["DAYTIME"],
    latent_heat_vaporization=results_physics_calib["LATENT_HEAT_VAPORIZATION_J_UMOL"],
)

scaling_factors_df = optimizer.run()
elapsed = time.time() - tic

print(f"\n[OK] Optimization completed in {elapsed:.1f}s")
print(f"  Scaling factors table: {len(scaling_factors_df)} rows")
print(f"    - Daytime bins: {(scaling_factors_df['DAYTIME'] == 1.0).sum()}")
print(f"    - Nighttime bins: {(scaling_factors_df['DAYTIME'] == 0.0).sum()}")
print(f"  Bootstrap runs: {optimizer.n_bootstrap}")

optimizer.stats()

# Plot optimization results
print(f"\nGenerating optimization results plots...")
optimizer.plot()

# %%
# Step 3a: Validate table on parallel data (Calibration Period)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ScopApplicator uses the optimized table to correct the calibration period data.
# Results should closely match the closed-path reference.

print(f"\n" + "=" * 80)
print("Step 3a: Validation on Parallel Data (Calibration Period)")
print("=" * 80)

tic = time.time()

applicator_calib = ScopApplicator(
    flux_type="CO2",
    fct_unsc=results_physics_calib["FCT_UNSC_gfRF"],
    scaling_factors_df=scaling_factors_df,
    flux_openpath=df_calibration["NEE_L3.1_L3.2_QCF_IRGA75"].copy(),
    classvar=df_calibration["USTAR_IRGA72"].copy(),
    daytime=results_physics_calib["DAYTIME"].copy()
)

applicator_calib.run()
results_calib_df = applicator_calib.get_results()
elapsed = time.time() - tic

print(f"\n[OK] Correction applied in {elapsed:.1f}s")

flux_op_uncorr = df_calibration["NEE_L3.1_L3.2_QCF_IRGA75"]
flux_op_corr = results_calib_df['NEE_OP_CORR']
flux_cp_ref = df_calibration["NEE_L3.1_L3.2_QCF_IRGA72"]

print(f"\nFlux statistics (calibration period):")
print(f"  Open-path (uncorrected): {flux_op_uncorr.mean():>8.3f} µmol m-2 s-1  (N={flux_op_uncorr.count()})")
print(f"  Open-path (corrected):   {flux_op_corr.mean():>8.3f} µmol m-2 s-1")
print(f"  Closed-path (reference): {flux_cp_ref.mean():>8.3f} µmol m-2 s-1  (N={flux_cp_ref.count()})")

print(f"\nTable validation statistics:")
applicator_calib.stats(flux_closedpath=flux_cp_ref)

# Plot correction validation dashboard
print(f"\nGenerating correction validation dashboard...")
applicator_calib.plot_dashboard(flux_closedpath=flux_cp_ref)

# %%
# Scaling Factors Lookup Table Details
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The scaling factors table is the key output of the calibration phase.
# Each bin shows the median scaling factor (SF_MEDIAN) and uncertainty ranges
# (quartiles and percentiles). The table is indexed by USTAR class and day/night.
#
# Key patterns:
# - Daytime: Higher scaling factors (0.13-0.38) due to stronger heating during high radiation
# - Nighttime: Lower scaling factors (0.02-0.09) due to weaker heating in darkness
# - Low USTAR: Larger factors (poor wind mixing accumulates heating effect)
# - High USTAR: Smaller factors (better wind mixing reduces heating effect)

print(f"\n" + "=" * 80)
print("SCALING FACTORS LOOKUP TABLE")
print("=" * 80)

print(f"\nTable structure: {len(scaling_factors_df)} rows × {len(scaling_factors_df.columns)} columns")
print(f"  Daytime bins: {(scaling_factors_df['DAYTIME'] == 1.0).sum()} (20 USTAR classes)")
print(f"  Nighttime bins: {(scaling_factors_df['DAYTIME'] == 0.0).sum()} (20 USTAR classes)")

print(f"\nMedian scaling factors:")
dt_sf = scaling_factors_df[scaling_factors_df['DAYTIME'] == 1.0]['SF_MEDIAN']
nt_sf = scaling_factors_df[scaling_factors_df['DAYTIME'] == 0.0]['SF_MEDIAN']
print(f"  Daytime median:   {dt_sf.median():.4f} (range: {dt_sf.min():.4f} - {dt_sf.max():.4f})")
print(f"  Nighttime median: {nt_sf.median():.4f} (range: {nt_sf.min():.4f} - {nt_sf.max():.4f})")

print(f"\nDaytime scaling factors (sample of first 10 bins):")
dt_sample = scaling_factors_df[scaling_factors_df['DAYTIME'] == 1.0][['GROUP_CLASSVAR', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR_MAX', 'SF_MEDIAN', 'SF_Q25', 'SF_Q75']].head(10)
for idx, row in dt_sample.iterrows():
    iqr = row['SF_Q75'] - row['SF_Q25']
    print(f"  Bin {int(row['GROUP_CLASSVAR']):2d} (USTAR {row['GROUP_CLASSVAR_MIN']:5.3f}-{row['GROUP_CLASSVAR_MAX']:5.3f}): {row['SF_MEDIAN']:.4f} ± {iqr:.4f}")

print(f"\nFull scaling factors table (all 40 bins):")
print(scaling_factors_df[['DAYTIME', 'GROUP_CLASSVAR', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR_MAX', 'SF_MEDIAN', 'SF_Q25', 'SF_Q75']].to_string(index=False))

# %%
# PHASE 2: APPLICATION — Correct Long-Term Data
# ==============================================
#
# Now apply the scaling factors table to long-term open-path flux data
# (without parallel measurements). The same physics model is used, but
# correction is applied using the pre-computed lookup table.

print(f"\n" + "=" * 80)
print("PHASE 2: APPLICATION — Correct Long-Term Flux Data")
print("=" * 80)

# Load long-term data
df_longterm = load_exampledata_parquet_lae()

print(f"\nLong-term data period: {df_longterm.index.min()} to {df_longterm.index.max()}")
print(f"Total records: {len(df_longterm):,}")
print("Note: For this example, using same dataset. In practice, this would be multi-year data.")

# %%
# Step 1b: Physics calculation (Long-term Period)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Apply ScopPhysics to the long-term dataset using the same methodology as calibration.

print(f"\n" + "=" * 80)
print("Step 1b: Physics Calculation (Long-term Period)")
print("=" * 80)

tic = time.time()

physics_longterm = ScopPhysics(
    flux_type="CO2",
    ta=df_longterm["TA_T1_47_1_gfXG_IRGA72"].copy(),
    gas_density=df_longterm["CO2_MOLAR_DENSITY_IRGA75"].copy() * 1000,
    rho_a=df_longterm["AIR_DENSITY_IRGA72"].copy(),
    rho_v=df_longterm["VAPOR_DENSITY_IRGA72"].copy(),
    u=df_longterm["U_IRGA72"].copy(),
    c_p=df_longterm["AIR_CP_IRGA72"].copy(),
    ustar=df_longterm["USTAR_IRGA72"].copy(),
    lat=47.478333,
    lon=8.364389,
    utc_offset=1,
)

physics_longterm.run(correction_method_base="JAR09", gapfill=True)
results_physics_longterm = physics_longterm.get_results()
elapsed = time.time() - tic

print(f"\n[OK] Physics calculation completed in {elapsed:.1f}s")
physics_longterm.stats()

# Plot diurnal cycles for long-term period
print(f"\nGenerating diurnal cycle plots...")
physics_longterm.plot_diel_cycles()

# %%
# Step 2b: Apply lookup table (Long-term Period)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ScopApplicator with the pre-computed scaling factors table to correct
# long-term fluxes. No optimization is needed; the table provides scaling factors.

print(f"\n" + "=" * 80)
print("Step 2b: Apply Scaling Factors Table (Long-term Period)")
print("=" * 80)
print("Correcting long-term fluxes using pre-computed lookup table...")

tic = time.time()

applicator_longterm = ScopApplicator(
    flux_type="CO2",
    fct_unsc=results_physics_longterm["FCT_UNSC_gfRF"],
    scaling_factors_df=scaling_factors_df,                   # Reuse table from calibration
    flux_openpath=df_longterm["NEE_L3.1_L3.2_QCF_IRGA75"].copy(),
    classvar=df_longterm["USTAR_IRGA72"].copy(),
    daytime=results_physics_longterm["DAYTIME"].copy()
)

applicator_longterm.run()
results_longterm_df = applicator_longterm.get_results()
elapsed = time.time() - tic

print(f"\n[OK] Correction applied in {elapsed:.1f}s")

# %%
# Long-Term Correction Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Examine the correction impact on the long-term dataset. The scaling factors
# table applies systematic correction across all seasons and years.

print(f"\n" + "=" * 80)
print("Long-Term Correction Impact Summary")
print("=" * 80)

flux_longterm_uncorr = df_longterm["NEE_L3.1_L3.2_QCF_IRGA75"]
flux_longterm_corr = results_longterm_df['NEE_OP_CORR']

print(f"\nFlux statistics (long-term period):")
print(f"  Open-path (uncorrected): {flux_longterm_uncorr.mean():>8.3f} µmol m-2 s-1  (N={flux_longterm_uncorr.count()})")
print(f"  Open-path (corrected):   {flux_longterm_corr.mean():>8.3f} µmol m-2 s-1")
print(f"  Correction (mean):       {(flux_longterm_corr - flux_longterm_uncorr).mean():>8.3f} µmol m-2 s-1")

# Budget impact
budget_uncorr = flux_longterm_uncorr.sum()
budget_corr = flux_longterm_corr.sum()
budget_change = budget_corr - budget_uncorr

print(f"\nAnnual budget impact (cumulative):")
print(f"  Uncorrected sum:  {budget_uncorr:>12,.0f} µmol m-2")
print(f"  Corrected sum:    {budget_corr:>12,.0f} µmol m-2")
print(f"  Net adjustment:   {budget_change:>12,.0f} µmol m-2  ({(budget_change/abs(budget_uncorr))*100:+.1f}%)")

print(f"\nDetailed correction statistics:")
applicator_longterm.stats()

# Plot long-term correction dashboard
print(f"\nGenerating long-term correction dashboard...")
applicator_longterm.plot_dashboard()

# %%
# Scaling factor assignment distribution (Long-term)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display how scaling factors were assigned across USTAR bins for the long-term period.

print(f"\n" + "=" * 80)
print("Scaling Factor Assignment (Long-term)")
print("=" * 80)

sf_col = results_longterm_df['SF']
group_col = results_longterm_df['GROUP_CLASSVAR']

print(f"\nScaling factor statistics by USTAR class:")
for daytime_flag, label in [(1.0, "DAYTIME"), (0.0, "NIGHTTIME")]:
    mask = results_physics_longterm['DAYTIME'] == daytime_flag
    if mask.sum() > 0:
        print(f"\n{label}:")
        for bin_id in range(10):  # Show first 10 bins as example
            bin_mask = (group_col == bin_id) & mask
            if bin_mask.sum() > 0:
                sf_vals = sf_col[bin_mask]
                print(f"  Bin {int(bin_id):2d}: SF={sf_vals.mean():.4f} ± {sf_vals.std():.4f} (N={bin_mask.sum()})")

# %%
# Using the Scaling Factors Table in Future Applications
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The scaling factors table is now ready for operational use. It can be saved
# and applied to any future period at the same site as long as the instrument
# configuration remains unchanged.
#
# .. code-block:: python
#
#     # Save table for future use
#     scaling_factors_df.to_csv("scaling_factors_table.csv", index=False)
#
#     # Later: Load and apply to new data period
#     sf_table = pd.read_csv("scaling_factors_table.csv")
#
#     applicator = ScopApplicator(
#         flux_type="CO2",
#         fct_unsc=fct_unsc_future,
#         scaling_factors_df=sf_table,
#         flux_openpath=flux_future,
#         classvar=ustar_future,
#         daytime=daytime_future
#     )
#     applicator.run()
#     corrected_flux = applicator.get_results()['NEE_OP_CORR']

print(f"\n" + "=" * 80)
print("PRODUCTION WORKFLOW COMPLETE")
print("=" * 80)

print(f"\n✅ Calibration phase: Scaling factors table created ({len(scaling_factors_df)} bins)")
print(f"✅ Application phase: Long-term fluxes corrected ({flux_longterm_corr.count()} records)")
print(f"\nScaling factors table ready for:")
print(f"  - Operational correction of multi-year flux datasets")
print(f"  - Annual carbon budget calculations")
print(f"  - Ecosystem response analysis")
print(f"  - Gap-filling and quality control workflows")

dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\nFinished. {dt_string}")
