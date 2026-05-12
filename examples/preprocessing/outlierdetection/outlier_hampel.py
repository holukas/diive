"""
====================
Hampel Filter
====================

Robust outlier detection using Median Absolute Deviation (MAD).
Detects values that deviate significantly from local trend, ideal for spike removal.
"""

# %%
# Load data and add synthetic noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a clean temperature series and add impulse noise to simulate spikes.
# Demonstrates the Hampel filter's ability to handle abrupt changes.

import warnings
import diive as dv

warnings.filterwarnings('ignore')

df = dv.load_exampledata_parquet()
s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()

# Add synthetic impulse noise
s_noise = dv.add_impulse_noise(
    series=s,
    factor_low=-10,
    factor_high=4,
    contamination=0.04,
    seed=42
)
s_noise.name = f"{s.name}+noise"

print("Test data created:")
print(f"  Total records: {len(s_noise)}")
print(f"  Range: {s_noise.min():.2f} to {s_noise.max():.2f}°C")
print(f"  Valid records: {s_noise.notna().sum()}")

# %%
# Apply Hampel filter with day/night separation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use separate thresholds for daytime and nighttime data.
# This accounts for different signal characteristics across the day.

ham_dtnt = dv.Hampel(
    series=s_noise,
    n_sigma=5.5,
    window_length=48 * 13,  # 13 days
    use_differencing=True,
    separate_day_night=True,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1
)
ham_dtnt.calc(repeat=False)

filtered_dtnt = ham_dtnt.filteredseries
flag_dtnt = ham_dtnt.flag

print("\nDay/night separation results:")
print(f"  Outliers detected: {(flag_dtnt == 2).sum()}")
print(f"  Valid records remaining: {filtered_dtnt.notna().sum()}")
print(f"  Filtered range: {filtered_dtnt.min():.2f} to {filtered_dtnt.max():.2f}°C")

# %%
# Apply global Hampel filter (single threshold)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use consistent threshold across entire series.
# Simpler, faster approach when time-of-day variation is not a concern.

ham_global = dv.Hampel(
    series=s_noise,
    n_sigma=5.5,
    window_length=48 * 13,
    use_differencing=True,
    separate_day_night=False,
    showplot=False,
    verbose=1
)
ham_global.calc(repeat=True)  # Iterate until convergence

filtered_global = ham_global.filteredseries
flag_global = ham_global.flag

print("\nGlobal threshold results:")
print(f"  Outliers detected: {(flag_global == 2).sum()}")
print(f"  Valid records remaining: {filtered_global.notna().sum()}")
print(f"  Filtered range: {filtered_global.min():.2f} to {filtered_global.max():.2f}°C")

# %%
# Comparison
# ^^^^^^^^^^
#
# Both approaches remove noise while preserving underlying trends.
# Day/night separation is useful when daytime and nighttime have different characteristics.

print("\nComparison:")
print(f"  Original valid: {s_noise.notna().sum()}")
print(f"  Day/night filtered: {filtered_dtnt.notna().sum()} ({100*filtered_dtnt.notna().sum()/s_noise.notna().sum():.1f}%)")
print(f"  Global filtered: {filtered_global.notna().sum()} ({100*filtered_global.notna().sum()/s_noise.notna().sum():.1f}%)")
