"""
=====================================
Moving Point (MP) USTAR Detection
=====================================

Detect USTAR threshold using the Moving Point (MP) method with bootstrapping.

The Moving Point method (Papale et al., 2006) is the standard FLUXNET approach
for identifying the friction velocity (u*) threshold below which eddy covariance
flux measurements become unreliable due to insufficient turbulent mixing.

Algorithm stratifies nighttime data by season and temperature, then by friction
velocity to identify where net ecosystem respiration (NEE) stabilizes. Bootstrap
resampling provides uncertainty estimates.

Best for: Robust USTAR threshold estimation following FLUXNET standards
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet_lae()

print("=" * 80)
print("Moving Point (MP) USTAR Threshold Detection")
print("=" * 80)

print(f"\nData loaded: {len(df)} records")
print(f"Period: {df.index.min()} to {df.index.max()}")

# %%
# Data overview
# ^^^^^^^^^^^^^
#
# Inspect available USTAR and flux data for the analysis.

# Find USTAR and flux columns
ustar_col = [col for col in df.columns if 'USTAR' in col][0]
flux_col = [col for col in df.columns if 'NEE' in col and 'QCF' in col][0]

print(f"\nUsing columns:")
print(f"  USTAR: {ustar_col}")
print(f"  Flux: {flux_col}")

# Filter to valid data
valid_mask = df[flux_col].notna() & df[ustar_col].notna()
n_valid = valid_mask.sum()

print(f"\nData availability:")
print(f"  Total records: {len(df)}")
print(f"  Valid records: {n_valid}")
print(f"  Mean USTAR: {df[ustar_col].mean():.4f} m/s")
print(f"  USTAR range: {df[ustar_col].min():.4f} to {df[ustar_col].max():.4f} m/s")
print(f"  Mean flux: {df[flux_col].mean():.3f} µmol m-2 s-1")

# %%
# Detect USTAR threshold with Moving Point method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Initialize detector with DataFrame. Column names are auto-detected.
# Run detection across 4 seasons with 7 temperature classes per season.

print(f"\n" + "=" * 80)
print("Moving Point (MP) Detection (Papale et al., 2006)")
print("=" * 80)

detector = dv.UstarMovingPointDetection(
    df=df,
    ta_classes_count=7,  # Temperature stratification classes
    ustar_classes_count=20,  # USTAR stratification classes per temperature class
    bootstrapping_times=100,  # Bootstrap iterations for uncertainty
    verbose=1
)

# Run detection
detector.detect()

# %%
# Display seasonal results
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show detected thresholds for each season.

print(f"\n" + detector.summary())

# %%
# Annual threshold and bootstrap uncertainty
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get conservative annual threshold and bootstrap confidence intervals.

annual_thresholds = detector.get_annual_thresholds()

print(f"\nAnnual thresholds (conservative approach - maximum across seasons):")
print(f"  Forward mode: {annual_thresholds['forward_mode']:.4f} m/s")
print(f"  Back mode: {annual_thresholds['back_mode']:.4f} m/s")

# %%
# Bootstrap uncertainty estimation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Run bootstrap resampling (100 iterations) to estimate confidence intervals.

print(f"\n" + "=" * 80)
print("Bootstrap Uncertainty Estimation")
print("=" * 80)

bootstrap_stats = detector.bootstrap(n_iter=100)

print(f"\nBootstrap results (100 iterations):")
print(bootstrap_stats)

print("\n[OK] Moving Point USTAR detection complete.")
