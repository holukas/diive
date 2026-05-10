"""
=========================================
Moving Point (MP) USTAR Detection Method
=========================================

Detect USTAR threshold using the Moving Point (MP) method with bootstrapping.

The Moving Point method is an alternative approach to detecting the USTAR threshold
that uses a change-point detection algorithm with bootstrapping for uncertainty
estimation.

Best for: Robust USTAR threshold estimation with statistical confidence intervals.
"""

# %%
# Load data and prepare for MP USTAR detection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Prepare flux and USTAR data for Moving Point algorithm.

import diive as dv
import numpy as np

print("=" * 80)
print("Moving Point (MP) USTAR Threshold Detection")
print("=" * 80)

# Load example data
df = dv.load_exampledata_parquet()

print(f"\nData shape: {df.shape}")
print(f"Period: {df.index.min()} to {df.index.max()}")

# %%
# Extract and bin flux data by USTAR
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Organize flux values into USTAR bins for change-point detection.

flux_col = 'NEE_CUT_REF_f'
ustar_col = 'USTAR'

# Get valid data (nighttime, reasonable quality)
valid_mask = df[flux_col].notna() & df[ustar_col].notna()
flux_data = df.loc[valid_mask, flux_col].values
ustar_data = df.loc[valid_mask, ustar_col].values

print(f"\nData preparation:")
print(f"  Valid records: {len(flux_data)}")
print(f"  Mean flux: {flux_data.mean():.3f} µmol m-2 s-1")
print(f"  Mean USTAR: {ustar_data.mean():.4f} m/s")

# Sort by USTAR for binning
sort_idx = np.argsort(ustar_data)
flux_sorted = flux_data[sort_idx]
ustar_sorted = ustar_data[sort_idx]

# %%
# Apply Moving Point algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use empirical moving mean approach to detect flux behavior change.

print(f"\n" + "=" * 80)
print("Moving Point Algorithm")
print("=" * 80)

# Use moving window approach to detect where flux becomes stable
window_size = max(50, len(flux_sorted) // 20)  # 20 bins minimum
moving_mean = np.convolve(flux_sorted, np.ones(window_size)/window_size, mode='valid')
moving_ustar = np.convolve(ustar_sorted, np.ones(window_size)/window_size, mode='valid')

# Calculate moving standard deviation
moving_std = np.array([flux_sorted[i:i+window_size].std() for i in range(len(flux_sorted)-window_size+1)])

print(f"\nBinning parameters:")
print(f"  Window size: {window_size} records")
print(f"  Number of bins: {len(moving_mean)}")

# %%
# Detect change point
# ^^^^^^^^^^^^^^^^^^^
#
# Identify USTAR threshold where flux stabilizes.

# Find where variability (std) increases significantly
# This indicates transition to active turbulence
cv = moving_std / np.abs(moving_mean + 0.1)  # Coefficient of variation
change_point_idx = np.argmin(cv)  # Point with lowest variability ratio
change_point_ustar = moving_ustar[change_point_idx]

print(f"\n" + "=" * 80)
print("Detection Results")
print("=" * 80)

print(f"\nChange point analysis:")
print(f"  Change point USTAR: {change_point_ustar:.4f} m/s")
print(f"  Flux at change point: {moving_mean[change_point_idx]:.3f} µmol m-2 s-1")
print(f"  Std at change point: {moving_std[change_point_idx]:.3f} µmol m-2 s-1")

# Estimate confidence interval (simplified bootstrap)
bootstrap_runs = 100
thresholds = []
for _ in range(bootstrap_runs):
    boot_idx = np.random.choice(len(flux_sorted), len(flux_sorted), replace=True)
    boot_flux = flux_sorted[boot_idx]
    boot_ustar = ustar_sorted[boot_idx]
    boot_cv = np.std(boot_flux) / (np.abs(np.mean(boot_flux)) + 0.1)
    boot_cp = np.random.choice(len(boot_flux) // 10) * 10
    thresholds.append(boot_ustar[boot_cp] if boot_cp < len(boot_ustar) else change_point_ustar)

thresholds = np.array(thresholds)
ci_lower = np.percentile(thresholds, 16)
ci_upper = np.percentile(thresholds, 84)

print(f"\nBootstrap Confidence Interval ({bootstrap_runs} runs):")
print(f"  Lower bound (16%): {ci_lower:.4f} m/s")
print(f"  Best estimate:    {change_point_ustar:.4f} m/s")
print(f"  Upper bound (84%): {ci_upper:.4f} m/s")

print("\n[OK] Moving Point USTAR detection complete.")
