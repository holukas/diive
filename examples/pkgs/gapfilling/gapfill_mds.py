"""
===================================
Marginal Distribution Sampling (MDS)
===================================

Gap-fill time series using Marginal Distribution Sampling.

MDS fills gaps by using the average flux value during similar meteorological
conditions (radiation, temperature, vapor pressure deficit). Uses a hierarchical
quality-based approach with progressively relaxed meteorological similarity windows.

Best for: Ecosystem flux gap-filling when ML training data is unavailable or when
physical similarity-based reconstruction is preferred over statistical learning.

References:
- Reichstein et al. (2005). On the separation of net ecosystem exchange into
  assimilation and ecosystem respiration. Global Change Biology, 11(9), 1424-1439.
  https://doi.org/10.1111/j.1365-2486.2005.001002.x
- Vekuri et al. (2023). A widely-used eddy covariance gap-filling method creates
  systematic bias in carbon balance estimates. Scientific Reports, 13(1), 1720.
  https://doi.org/10.1038/s41598-023-28827-2
"""

# %%
# Overview: Marginal Distribution Sampling (MDS)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MDS is an unsupervised gap-filling method that finds periods with similar
# meteorological conditions and uses their mean flux as the gap-filled value.
#
# Why MDS?
# - No training/validation data required (unsupervised)
# - Fast execution
# - Physically meaningful (based on meteorological similarity)
# - No overfitting risk
# - Widely used in FLUXNET, ICOS, REddyProc
#
# How it works:
# 1. For each gap, look for similar meteorological conditions
# 2. Similarity defined by: radiation (Rg), temperature (Ta), VPD
# 3. Hierarchical approach: start strict (7 days), relax to 140+ days
# 4. Fill gap with mean flux from similar periods
#
# Quality levels (1-26):
# - Levels 1-3: High quality (7-14 days, all 3 variables)
# - Levels 4-5: Diurnal cycle (1-2 hours)
# - Levels 6+: Progressively lower quality as window expands

import time

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

# Example: Use full year 2022 (realistic scenario with large gaps)
# Note: Takes longer to process; comment out to use July subset below
print("Loading example data (2022)...")
df_full_year = df.loc[df.index.year == 2022].copy()

# Select high-quality data only (filter by QCF flag)
print("Filtering data by quality flag (QCF=0 only)...")
target_col = 'NEE_CUT_REF_orig'
df_full_year.loc[df_full_year['QCF_NEE'] > 0, target_col] = np.nan

print(f"Data summary:")
print(f"  Period: {df_full_year.index.min().date()} to {df_full_year.index.max().date()}")
print(f"  Total records: {len(df_full_year)}")
print(f"  Available NEE: {df_full_year[target_col].notna().sum()}")
print(
    f"  Missing NEE: {df_full_year[target_col].isna().sum()} ({100 * df_full_year[target_col].isna().sum() / len(df_full_year):.1f}%)")

# Use faster subset for demonstration (comment this out for full-year analysis)
print("\nUsing July subset for faster execution (comment out for full year)...")
df = df_full_year.loc[(df_full_year.index.month == 7)].copy()

print(f"Using data: {len(df)} records")
print(f"  Available NEE: {df[target_col].notna().sum()}")
print(f"  Missing NEE: {df[target_col].isna().sum()}")

# %%
# MDS setup and configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Configure meteorological variables and similarity tolerances.
# CRITICAL: Variables must be in correct units!

# Flux and meteorological variables
flux_col = 'NEE_CUT_REF_orig'  # Flux variable to gap-fill
ta_col = 'Tair_f'  # Air temperature (in C)
swin_col = 'Rg_f'  # Short-wave radiation (in W/m2)
vpd_col = 'VPD_f'  # Vapor pressure deficit (in hPa in this dataset)

# MDS tolerance parameters
swin_tol = [20, 50]  # W/m2: low radiation 20, high radiation 50
ta_tol = 2.5  # C: temperature tolerance
vpd_tol = 0.5  # kPa: VPD tolerance (convert from hPa)
avg_min_n_vals = 5  # Min flux values needed to calculate average

print("MDS Configuration:")
print(f"  Flux column: {flux_col}")
print(f"  Ta tolerance: +or- {ta_tol} C")
print(f"  Rg tolerance: low={swin_tol[0]}, high={swin_tol[1]} W/m2")
print(f"  VPD tolerance: +or- {vpd_tol} kPa")
print(f"  Min values for average: {avg_min_n_vals}")

# Convert VPD from hPa to kPa
df[vpd_col] = df[vpd_col] * 0.1

# %%
# Run MDS gap-filling
# ^^^^^^^^^^^^^^^^^^^

print("\nStarting MDS gap-filling...")
start_time = time.perf_counter()

mds = dv.FluxMDS(
    df=df,
    flux=flux_col,
    ta=ta_col,
    swin=swin_col,
    vpd=vpd_col,
    swin_tol=swin_tol,
    ta_tol=ta_tol,
    vpd_tol=vpd_tol,
    avg_min_n_vals=avg_min_n_vals,
    verbose=1
)

mds.run()
elapsed_time = time.perf_counter() - start_time

print(f"Execution time: {elapsed_time:.2f} seconds")

# %%
# Results: Gap-filling report
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 70)
print("MDS GAP-FILLING RESULTS")
print("=" * 70)

mds.report()

# %%
# Results: Performance scores
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Scores computed by cross-validation on measured flux data
# (estimate of gap-filling accuracy)

print("\nPerformance Scores (from cross-validation):")
print(f"  R2: {mds.scores_['r2']:.4f}")
print(f"  RMSE: {mds.scores_['rmse']:.4f} (root mean squared error)")
print(f"  MAE: {mds.scores_['mae']:.4f} (mean absolute error)")
print(f"  Median AE: {mds.scores_['medae']:.4f}")
print(f"  Max Error: {mds.scores_['maxe']:.4f}")
print(f"  Mean quality flag: {mds.scores_['mean_quality_flag_gap_predictions']:.2f} (lower=better)")

# %%
# Results: Quality flag analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Understand how many gaps were filled at each quality level

gf_df = mds.gapfilling_df_
flag_col = [c for c in gf_df.columns if 'FLAG' in c and 'FILLED' in c][0]

print("\nGap-filling quality distribution:")
quality_counts = gf_df[flag_col].value_counts().sort_index()
for quality_level, count in quality_counts.items():
    if quality_level == 0:
        label = "Measured (not gap-filled)"
    else:
        label = f"Quality level {int(quality_level)}"
    percent = 100 * count / len(gf_df)
    print(f"  {label:.<40} {count:>6} ({percent:>5.1f}%)")

# %%
# Visualization: Time series comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Compare original (with gaps) vs gap-filled

print("\nGenerating visualizations...")

# Get original and gap-filled series
original_col = flux_col
gapfilled_col = mds.target_gapfilled

# Plot time series
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(gf_df.index, gf_df[original_col], 'k-', alpha=0.5, linewidth=1, label='Measured')
ax.plot(gf_df.index, gf_df[gapfilled_col], 'b-', alpha=0.7, linewidth=1, label='Gap-filled')
ax.set_xlabel('Date')
ax.set_ylabel('NEE (umol/m2/s)')
ax.set_title('MDS Gap-Filling: Measured vs Gap-Filled NEE')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.show()

# %%
# Visualization: Cumulative flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(gf_df.index, gf_df[original_col].cumsum(), 'k-', alpha=0.5, linewidth=1.5, label='Measured (incomplete)')
ax.plot(gf_df.index, gf_df[gapfilled_col].cumsum(), 'b-', alpha=0.7, linewidth=1.5, label='Gap-filled (complete)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative NEE (umol/m2/s)')
ax.set_title('Cumulative Flux: Impact of Gap-Filling')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.show()

# %%
# Visualization: Heatmap comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

hm_original = dv.HeatmapDateTime(series=gf_df[original_col])
hm_original.plot(
    ax=axes[0],
    title='Original NEE (with gaps)'
)
axes[0].set_xlabel('Hour of day')
axes[0].set_ylabel('Month')

hm_gapfilled = dv.HeatmapDateTime(series=gf_df[gapfilled_col])
hm_gapfilled.plot(
    ax=axes[1],
    title='Gap-filled NEE (complete)'
)
axes[1].set_xlabel('Hour of day')
axes[1].set_ylabel('Month')

plt.tight_layout()
fig.show()

print("\n[OK] MDS gap-filling demonstration complete")
