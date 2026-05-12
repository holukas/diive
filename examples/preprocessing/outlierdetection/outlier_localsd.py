"""
=========================
Local Standard Deviation
=========================

Adaptive outlier detection based on rolling window statistics.
Identifies values deviating significantly from local median and standard deviation.
"""

# %%
# Create test data with impulse noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate summer temperature data and add synthetic spikes.

import warnings
import diive as dv

warnings.filterwarnings('ignore')

df = dv.load_exampledata_parquet()
s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()
s = s.loc[s.index.month == 7].copy()

# Add synthetic impulse noise
s_noise = dv.add_impulse_noise(
    series=s,
    factor_low=-10,
    factor_high=3,
    contamination=0.04,
    seed=42
)
s_noise.name = f"{s.name}+noise"

print("Test data created:")
print(f"  Records: {len(s_noise)}")
print(f"  Valid: {s_noise.notna().sum()}")
print(f"  Range: {s_noise.min():.2f} to {s_noise.max():.2f}°C")

# %%
# Day/Night separated approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use separate window sizes and thresholds for daytime and nighttime.
# Adapts to different signal characteristics across the day.

lsd_dtnt = dv.LocalSD(
    series=s_noise,
    separate_daytime_nighttime=True,
    n_sd=[3, 2],
    winsize=[48 * 2, 48 * 1],  # 2-day daytime window, 1-day nighttime
    constant_sd=False,
    lat=46.0,
    lon=11.0,
    utc_offset=1,
    showplot=False,
    verbose=1
)
lsd_dtnt.calc(repeat=True)

flag_dtnt = lsd_dtnt.get_flag()
filtered_dtnt = s_noise.copy()
filtered_dtnt.loc[flag_dtnt == 2] = None

print("\nDay/night separated results:")
print(f"  Outliers detected: {(flag_dtnt == 2).sum()}")
print(f"  Valid records: {filtered_dtnt.notna().sum()}")
print(f"  Filtered range: {filtered_dtnt.min():.2f} to {filtered_dtnt.max():.2f}°C")

# %%
# Global threshold approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Single window size and standard deviation threshold applied to all records.
# Simpler computation, faster execution.

lsd_global = dv.LocalSD(
    series=s_noise,
    n_sd=2,
    winsize=48 * 2,  # 2-day rolling window
    constant_sd=True,  # Global SD instead of rolling
    showplot=False,
    verbose=1
)
lsd_global.calc(repeat=True)

flag_global = lsd_global.get_flag()
filtered_global = s_noise.copy()
filtered_global.loc[flag_global == 2] = None

print("\nGlobal threshold results:")
print(f"  Outliers detected: {(flag_global == 2).sum()}")
print(f"  Valid records: {filtered_global.notna().sum()}")
print(f"  Filtered range: {filtered_global.min():.2f} to {filtered_global.max():.2f}°C")

# %%
# Comparison
# ^^^^^^^^^^
#
# Both methods effectively remove spikes while preserving underlying trends.

print("\n" + "="*50)
print("Method Comparison")
print("="*50)
print(f"Original valid: {s_noise.notna().sum()}")
print(f"Day/night filtered: {filtered_dtnt.notna().sum()} ({100*filtered_dtnt.notna().sum()/s_noise.notna().sum():.1f}%)")
print(f"Global filtered: {filtered_global.notna().sum()} ({100*filtered_global.notna().sum()/s_noise.notna().sum():.1f}%)")
