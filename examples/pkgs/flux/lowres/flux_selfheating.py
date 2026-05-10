"""
=========================================
Self-Heating Correction (SCOP Methodology)
=========================================

Remove spurious CO2 flux caused by sun-induced heating of open-path IRGA sensors.

Demonstrates SCOP (Self-heating Correction for Open-Path) methodology to remove
spurious CO2 flux measurements caused by sun-induced heating of instrument surfaces.

Includes physics modeling, optimization of scaling factors, and correction application.

Best for: Correcting systematic bias in open-path IRGA CO2 flux measurements.
"""

# %%
# Self-heating correction workflow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The SCOP methodology uses three steps:
# 1. ScopPhysics: Calculate unscaled flux correction term (FCT_UNSC)
# 2. ScopOptimizer: Optimize scaling factors using closed-path reference
# 3. ScopApplicator: Apply final correction to open-path flux

from diive.pkgs.flux.lowres.selfheating import ScopPhysics, ScopOptimizer, ScopApplicator
from diive.configs.exampledata import load_exampledata_parquet_lae
import time

print("=" * 80)
print("SCOP Self-Heating Correction Workflow")
print("=" * 80)

# Load example data with parallel IRGA measurements
df = load_exampledata_parquet_lae()

# Parallel measurements starting 27 May 2016
df = df.loc["2016-05-27 00:15:00":"2017-12-11 23:45:00"].copy()

print(f"\nData period: {df.index.min()} to {df.index.max()}")
print(f"Total records: {len(df)}")

# %%
# Step 1: Calculate physics-based correction term
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ScopPhysics to calculate the unscaled flux correction term (FCT_UNSC)
# based on radiation, temperature, and wind speed.

print(f"\n" + "=" * 80)
print("Step 1: Physics-Based Correction Term Calculation")
print("=" * 80)

tic = time.time()

physics = ScopPhysics(
    flux_type="CO2",
    ta=df["TA_T1_47_1_gfXG_IRGA72"].copy(),
    gas_density=df["CO2_MOLAR_DENSITY_IRGA75"].copy() * 1000,  # Convert to umol m-3
    rho_a=df["AIR_DENSITY_IRGA72"].copy(),
    rho_v=df["VAPOR_DENSITY_IRGA72"].copy(),
    u=df["U_IRGA72"].copy(),
    c_p=df["AIR_CP_IRGA72"].copy(),
    ustar=df["USTAR_IRGA72"].copy(),
    lat=47.478333,  # CH–LAE
    lon=8.364389,   # CH–LAE
    utc_offset=1,
)
physics.run(correction_method_base="JAR09", gapfill=True)
physics.stats()

results_physics_df = physics.get_results()
elapsed = time.time() - tic

print(f"\n[OK] Physics calculation completed in {elapsed:.1f}s")
print(f"  Correction term (FCT_UNSC) calculated")
print(f"  Gap-filling with Random Forest applied")

# %%
# Step 2: Optimize scaling factors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ScopOptimizer to find optimal scaling factors by comparing
# open-path and closed-path measurements.

print(f"\n" + "=" * 80)
print("Step 2: Scaling Factor Optimization")
print("=" * 80)

tic = time.time()

optimizer = ScopOptimizer(
    flux_type="CO2",
    fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
    class_var=df["USTAR_IRGA72"].copy(),
    n_classes=5,
    n_bootstrap_runs=5,
    flux_openpath=df["NEE_L3.1_L3.2_QCF_IRGA75"].copy(),
    flux_closedpath=df["NEE_L3.1_L3.2_QCF_IRGA72"].copy(),
    daytime=results_physics_df["DAYTIME"],
    latent_heat_vaporization=results_physics_df["LATENT_HEAT_VAPORIZATION_J_UMOL"],
)
scaling_factors_df = optimizer.run()
optimizer.stats()

elapsed = time.time() - tic

print(f"\n[OK] Optimization completed in {elapsed:.1f}s")
print(f"  Scaling factors determined for {len(scaling_factors_df)} USTAR classes")
print(f"  {optimizer.n_bootstrap_runs} bootstrap runs executed")

# %%
# Step 3: Apply correction
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ScopApplicator to apply the optimized correction to the flux data.

print(f"\n" + "=" * 80)
print("Step 3: Correction Application")
print("=" * 80)

tic = time.time()

applicator = ScopApplicator(
    flux_type="CO2",
    fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
    scaling_factors_df=scaling_factors_df,
    flux_openpath=df["NEE_L3.1_L3.2_QCF_IRGA75"].copy(),
    classvar=df["USTAR_IRGA72"].copy(),
    daytime=results_physics_df["DAYTIME"].copy()
)
applicator.run()
corrected_flux = applicator.get_corrected_flux()

elapsed = time.time() - tic

print(f"\n[OK] Correction applied in {elapsed:.1f}s")
print(f"  Original flux (open-path): {df['NEE_L3.1_L3.2_QCF_IRGA75'].mean():.3f} µmol m-2 s-1")
print(f"  Corrected flux: {corrected_flux.mean():.3f} µmol m-2 s-1")

print("\n[OK] Self-heating correction workflow complete.")
