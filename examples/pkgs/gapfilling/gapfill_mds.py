"""
===================================
Marginal Distribution Sampling (MDS)
===================================

Gap-fill time series using Marginal Distribution Sampling.

MDS fills gaps by using the average flux value during similar meteorological
conditions (radiation, temperature, vapor pressure deficit). Uses a hierarchical
quality-based approach with progressively relaxed meteorological similarity windows.

Reference: Reichstein et al (2005)
https://doi.org/10.1111/j.1365-2486.2005.001002.x
"""

# %%
# Marginal Distribution Sampling (MDS) gap-filling
# =================================================
#
# This example demonstrates MDS gap-filling which replaces missing flux values
# with average flux from similar meteorological conditions.
#
# MDS uses a hierarchical quality approach:
# - Quality A (A1-A3): High quality, all variables within 7-14 days
# - Quality B (B1-B4): Medium quality, variables within 21-28 days
# - Quality C+: Low quality, variables within 35-140+ days
#
# Advantages:
# - No training data required (unsupervised)
# - Fast execution
# - Physically meaningful (based on meteorological similarity)
# - No overfitting risk
#
# Disadvantages:
# - May underestimate complex non-linear relationships
# - Requires sufficient meteorological variability in the dataset

import time
import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

# Use July 2022 for faster example execution
df = df.loc[(df.index.year == 2022) & (df.index.month == 7)].copy()

print(f"Data loaded: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values: {df['NEE_CUT_REF_orig'].isnull().sum()}")

# %%
# MDS setup and configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Configure the meteorological variables and similarity tolerances.
# **CRITICAL:** Variables must be in the correct units.

# Variables
flux = 'NEE_CUT_REF_orig'  # Flux to gap-fill
ta = 'Tair_f'             # Air temperature (must be °C)
swin = 'Rg_f'             # Short-wave incoming radiation (must be W/m²)
vpd = 'VPD_f'             # Vapor pressure deficit (in this file: hPa)

# MDS tolerance settings
swin_tol = [20, 50]  # W m-2 (low radiation: 20, high: 50)
ta_tol = 2.5         # °C
vpd_tol = 0.5        # kPa (will convert from hPa)
avg_min_n_vals = 5   # Minimum flux values to calculate average

# Convert VPD from hPa to kPa
df[vpd] = df[vpd].multiply(0.1)

# %%
# Run MDS gap-filling
# ^^^^^^^^^^^^^^^^^^^

start_time = time.perf_counter()

mds = dv.FluxMDS(
    df=df,
    flux=flux,
    ta=ta,
    swin=swin,
    vpd=vpd,
    swin_tol=swin_tol,
    ta_tol=ta_tol,
    vpd_tol=vpd_tol,
    avg_min_n_vals=avg_min_n_vals,
    verbose=1
)

mds.run()
elapsed_time = time.perf_counter() - start_time

print(f"\nMDS gap-filling execution time: {elapsed_time:.2f} seconds")

# %%
# Results and visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^

mds.report()
mds.showplot()

print("✓ MDS gap-filling complete.")
