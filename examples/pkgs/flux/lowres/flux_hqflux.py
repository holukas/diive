"""
==================================================
High-Quality Flux Analysis with Hampel Filtering
==================================================

Robust outlier detection for CO2 flux using Hampel filter with day/night separation.

Demonstrates robust outlier detection for CO2 flux (NEE) using the Hampel filter
(Median Absolute Deviation) with automatic day/night separation based on
solar geometry. The Hampel method is ideal for removing measurement spikes
while preserving ecosystem signal.

Best for: Identifying and removing measurement spikes in CO2 flux data.
"""

# %%
# Load and prepare CO2 flux data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Load example eddy covariance data and select records with good quality flags.

from diive.configs.exampledata import load_exampledata_parquet_cha
from diive.pkgs.flux.lowres.hqflux import analyze_highest_quality_flux

df = load_exampledata_parquet_cha()
keeprows = df['FC_SSITC_TEST'] == 0
df = df[keeprows].copy()
flux_fc = df['FC'].copy()

print("=" * 80)
print("Data Summary")
print("=" * 80)
print(f"\nData: {flux_fc.name}")
print(f"Period: {flux_fc.index.min().date()} to {flux_fc.index.max().date()}")
print(f"Total records: {len(flux_fc)}")
print(f"Valid records: {flux_fc.count()}")
print(f"Missing: {flux_fc.isnull().sum()} ({flux_fc.isnull().sum() / len(flux_fc) * 100:.1f}%)")

# %%
# Apply Hampel filter with day/night separation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Run high-quality flux analysis using Hampel filter (Median Absolute Deviation).
# Automatically separates daytime and nighttime using solar elevation angle,
# with separate strictness thresholds for each period.

results = analyze_highest_quality_flux(
    flux=flux_fc,
    lat=47.286417,  # Swiss FluxNet site (CH-DAV)
    lon=8.010325,
    utc_offset=1,  # CET
    window_length=48 * 13,  # 13 days at 30-min frequency
    n_sigma_daytime=5.5,
    n_sigma_nighttime=5.5,
    use_differencing=True,  # Papale method: isolate spikes from trends
    showplot=True
)

# %%
# Examine results
# ^^^^^^^^^^^^^^^
#
# Inspect the filtered data and understand the Hampel filter method.

print(f"\n" + "=" * 80)
print("Results Summary")
print("=" * 80)
print(f"Filtered data shape: {results.shape}")
print(f"Columns: {list(results.columns)}")

print(f"\nInterpretation:")
print(f"  - Rolling median ± 3 SD shows expected CO2 flux range")
print(f"  - Points outside this range may indicate sensor issues or extreme events")

print(f"\nHampel Filter Method:")
print(f"  - Median Absolute Deviation (MAD) for robust statistics")
print(f"  - Double-differencing (Papale et al. 2006) to remove biological trends")
print(f"  - Automatic day/night thresholds based on solar elevation angle")

print("\n[OK] High-quality flux analysis complete.")
