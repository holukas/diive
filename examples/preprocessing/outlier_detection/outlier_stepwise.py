"""
============================
Sequential Outlier Detection
============================

Chain multiple detection methods to progressively filter outliers.
Each method operates on data filtered by previous methods, refining results.
"""

# %%
# Generate synthetic test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create monthly time series with trend, seasonality, noise, and spikes.
# Simulates realistic noisy measurement data.

import matplotlib.pyplot as plt

import diive as dv

df_noisy = dv.generate_noisy_timeseries(
    start_date='2024-01-01',
    periods=48 * 30,  # ~1 month half-hourly
    freq='30min',
    trend_slope=0.01,
    seasonal_strength=9,
    noise_level=2,
    outlier_fraction=0.1  # 10% spikes
)
df_noisy.index.name = 'TIMESTAMP_END'

print("Synthetic data generated:")
print(f"  Records: {len(df_noisy)}")
print(f"  Valid: {df_noisy['observed_value'].notna().sum()}")
print(f"  Period: {df_noisy.index.min()} to {df_noisy.index.max()}")

# %%
# Initialize stepwise detector
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create orchestrator that chains multiple detection methods.
# Each method filters data already cleaned by previous methods.

from diive.pkgs.preprocessing.outlier_detection import StepwiseOutlierDetection

detector = StepwiseOutlierDetection(
    dfin=df_noisy,
    col='observed_value',
    site_lat=46.8,
    site_lon=8.6,
    utc_offset=1
)

print("\nStepwise orchestrator initialized")
print(f"  Column: {detector.col}")
print(f"  Site lat/lon: {detector.site_lat}, {detector.site_lon}")

# %%
# Apply detection methods sequentially
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each test operates on data cleaned by previous tests.
# Order matters: start with aggressive filtering, then refine.

print("\n" + "=" * 60)
print("Sequential Outlier Detection Chain")
print("=" * 60)

# 1. Hampel filter
print("\n1. Hampel filter (MAD-based spike detection)...")
detector.flag_outliers_hampel_test(
    window_length=7 * 48,
    n_sigma_daytime=4.5,
    n_sigma_nighttime=4.5,
    use_differencing=True,
    separate_daytime_nighttime=True,
    showplot=False,
    verbose=False,
    repeat=True
)
print(f"   > Found {(detector.last_flag == 2).sum()} outliers")
detector.addflag()

# 2. LocalSD
print("\n2. LocalSD (local standard deviation)...")
detector.flag_outliers_localsd_test(
    n_sd=[3.5, 3.5],
    winsize=[24, 24],
    separate_daytime_nighttime=True,
    showplot=False,
    constant_sd=False,
    verbose=False,
    repeat=True
)
print(f"   > Found {(detector.last_flag == 2).sum()} additional outliers")
detector.addflag()

# 3. Z-score global
print("\n3. Z-score global threshold...")
detector.flag_outliers_zscore_test(
    thres_zscore=4,
    showplot=False,
    verbose=False,
    repeat=True
)
print(f"   > Found {(detector.last_flag == 2).sum()} additional outliers")
detector.addflag()

# 4. Z-score rolling
print("\n4. Z-score rolling window (adaptive)...")
detector.flag_outliers_zscore_rolling_test(
    thres_zscore=3.5,
    showplot=False,
    verbose=False,
    repeat=True,
    winsize=None
)
print(f"   > Found {(detector.last_flag == 2).sum()} additional outliers")
detector.addflag()

# 5. Z-score day/night
print("\n5. Z-score day/night separation...")
detector.flag_outliers_zscore_test(
    thres_zscore=4,
    separate_daytime_nighttime=True,
    lat=detector.site_lat,
    lon=detector.site_lon,
    utc_offset=detector.utc_offset,
    showplot=False,
    verbose=False,
    repeat=True
)
print(f"   > Found {(detector.last_flag == 2).sum()} additional outliers")
detector.addflag()

# 6. Increments
print("\n6. Increment-based detection (abrupt changes)...")
detector.flag_outliers_increments_zcore_test(
    thres_zscore=3.5,
    showplot=False,
    verbose=False,
    repeat=True
)
print(f"   > Found {(detector.last_flag == 2).sum()} additional outliers")
detector.addflag()

# %%
# View results
# ^^^^^^^^^^^^
#
# Summary of filtering effectiveness and detailed flag information.

cleaned = detector.series_hires_cleaned
original = detector.series_hires_orig

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)
print(f"Original valid records: {original.notna().sum()}")
print(f"Final valid records: {cleaned.notna().sum()}")
print(f"Total outliers removed: {(original.notna() & cleaned.isna()).sum()}")
print(f"Data retention: {100 * cleaned.notna().sum() / original.notna().sum():.1f}%")

print(f"\nFlag columns applied: {detector.flags.shape[1]}")
print(f"Sample flags (first 5 records):\n{detector.flags.head()}")

# %%
# Visualize original and cleaned data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Plot showing detected outliers (red) overlaid on original data,
# plus final cleaned series.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Top: Original with outliers marked
outliers_mask = original.notna() & cleaned.isna()
ax1.plot(original.index, original.values, label='Original data',
         color='#1f77b4', linewidth=1.2, alpha=0.8)
ax1.scatter(original[outliers_mask].index, original[outliers_mask].values,
            color='#d62728', s=50, label='Detected outliers', zorder=5, alpha=0.8)
ax1.set_title('Step 1: Original Data with Detected Outliers', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=11)
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3)

# Bottom: Cleaned data
ax2.plot(original.index, original.values, label='Original (reference)',
         color='#1f77b4', linewidth=0.8, alpha=0.3, linestyle='--')
ax2.plot(cleaned.index, cleaned.values, label='Cleaned data',
         color='#2ca02c', linewidth=1.5, marker='o', markersize=3, alpha=0.8)
ax2.fill_between(cleaned.index, cleaned.min() - 1, cleaned,
                 alpha=0.1, color='#2ca02c')
ax2.set_title('Step 2: Final Cleaned Series', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xlabel('Timestamp', fontsize=11)
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
