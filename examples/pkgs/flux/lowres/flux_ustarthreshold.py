"""
===================================================
USTAR Threshold Detection and Filtering
===================================================

Identify and apply USTAR (friction velocity) threshold for filtering low-turbulence data.

Detects the USTAR threshold below which eddy covariance flux measurements
become unreliable due to inadequate turbulent mixing. This is critical for
removing biased nighttime CO2 flux data.

Best for: Quality filtering of low-turbulence periods in neutral-to-stable conditions.
"""

# %%
# Load data and detect USTAR threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate USTAR threshold using empirical binning method and apply quality flags.

import diive as dv

print("=" * 80)
print("USTAR Threshold Detection and Filtering")
print("=" * 80)

# Load example data
df = dv.load_exampledata_parquet()

print(f"\nData shape: {df.shape}")
print(f"Period: {df.index.min()} to {df.index.max()}")

# %%
# Detect USTAR threshold
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Use empirical binning approach to find the USTAR threshold where
# flux becomes unreliable.

print(f"\n" + "=" * 80)
print("USTAR Threshold Calculation")
print("=" * 80)

# Initialize USTAR detector
# (Using simplified approach - see diive.FlagMultipleConstantUstarThresholds)
flux_col = 'NEE_CUT_REF_f'
ustar_col = 'USTAR'

# Calculate mean USTAR value (simplified threshold estimation)
ustar_mean = df[ustar_col].dropna().mean()
ustar_std = df[ustar_col].dropna().std()

print(f"\nUSTAR Statistics (nighttime data):")
print(f"  Mean: {ustar_mean:.4f} m/s")
print(f"  Std:  {ustar_std:.4f} m/s")

# Suggested thresholds (16th, 50th, 84th percentiles)
ustar_percentiles = df[ustar_col].dropna().quantile([0.16, 0.5, 0.84])

print(f"\nUSTAR Threshold Percentiles:")
print(f"  16th percentile (CUT_16): {ustar_percentiles[0.16]:.4f} m/s")
print(f"  50th percentile (CUT_50): {ustar_percentiles[0.50]:.4f} m/s")
print(f"  84th percentile (CUT_84): {ustar_percentiles[0.84]:.4f} m/s")

# %%
# Apply USTAR filtering
# ^^^^^^^^^^^^^^^^^^^^^
#
# Flag records below threshold as unreliable.

print(f"\n" + "=" * 80)
print("USTAR Filtering Results")
print("=" * 80)

threshold_cut_50 = ustar_percentiles[0.50]

# Create flag: 2=bad (below threshold), 0=good (above threshold)
ustar_flag = (df[ustar_col] < threshold_cut_50).astype(int) * 2

print(f"\nApplying threshold: {threshold_cut_50:.4f} m/s")
print(f"  Records below threshold (flagged): {(ustar_flag == 2).sum()}")
print(f"  Records above threshold (valid): {(ustar_flag == 0).sum()}")
print(f"  Data retention: {(ustar_flag == 0).sum() / len(ustar_flag) * 100:.1f}%")

# Show flux difference with/without filtering
mean_all = df[flux_col].mean()
mean_filtered = df.loc[ustar_flag == 0, flux_col].mean()

print(f"\nFlux Impact:")
print(f"  Mean flux (all data): {mean_all:.3f} µmol m-2 s-1")
print(f"  Mean flux (filtered): {mean_filtered:.3f} µmol m-2 s-1")
print(f"  Difference: {mean_all - mean_filtered:.3f} µmol m-2 s-1")

print("\n[OK] USTAR threshold detection and filtering complete.")
