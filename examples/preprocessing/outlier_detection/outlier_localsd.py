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
s_noise = dv.variables.add_impulse_noise(
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

lsd_dtnt = dv.outliers.LocalSD(
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

lsd_global = dv.outliers.LocalSD(
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

print("\n" + "=" * 50)
print("Method Comparison")
print("=" * 50)
print(f"Original valid: {s_noise.notna().sum()}")
print(
    f"Day/night filtered: {filtered_dtnt.notna().sum()} ({100 * filtered_dtnt.notna().sum() / s_noise.notna().sum():.1f}%)")
print(
    f"Global filtered: {filtered_global.notna().sum()} ({100 * filtered_global.notna().sum() / s_noise.notna().sum():.1f}%)")

# %%
# Parameter tuning: n_sd sensitivity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Vary the number of standard deviations (n_sd) to adjust outlier detection sensitivity.
# Lower n_sd → stricter (more outliers), higher n_sd → permissive (fewer outliers).

print("\n" + "=" * 50)
print("Parameter Tuning: n_sd Sensitivity")
print("=" * 50)

n_sd_values = [1.5, 2.0, 2.5, 3.0, 3.5]

for n_sd in n_sd_values:
    lsd = dv.outliers.LocalSD(
        series=s_noise,
        n_sd=n_sd,
        winsize=48 * 2,
        constant_sd=False,
        showplot=False,
        verbose=0
    )
    lsd.calc(repeat=False)

    flag = lsd.get_flag()
    outliers = (flag == 2).sum()
    retained = (flag != 2).sum()
    outlier_rate = 100 * outliers / s_noise.notna().sum()

    print(f"\nn_sd = {n_sd}:")
    print(f"  Outliers detected: {outliers} ({outlier_rate:.2f}%)")
    print(f"  Data retained: {retained} ({100 * retained / s_noise.notna().sum():.1f}%)")

# %%
# Parameter tuning: window size effect
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Shorter windows catch local anomalies, longer windows smooth over variations.

print("\n" + "=" * 50)
print("Parameter Tuning: Window Size")
print("=" * 50)

window_sizes = [48, 48 * 2, 48 * 5, 48 * 10]  # 1 to 10 days at 30-min resolution

for winsize in window_sizes:
    lsd_win = dv.outliers.LocalSD(
        series=s_noise,
        n_sd=2.5,
        winsize=winsize,
        constant_sd=False,
        showplot=False,
        verbose=0
    )
    lsd_win.calc(repeat=False)

    flag = lsd_win.get_flag()
    outliers = (flag == 2).sum()
    retained = (flag != 2).sum()

    days = winsize / 48
    print(f"\nWindow size = {winsize} steps ({days:.1f} days):")
    print(f"  Outliers detected: {outliers}")
    print(f"  Data retained: {retained}")

# %%
# Choosing parameters
# ^^^^^^^^^^^^^^^^^^^^
#
# **Guidelines for LocalSD parameter selection:**
#
# - **n_sd:** 2.0-2.5 for general use, 2.5-3.5 for sensitive data
# - **winsize:** 1-2 days for fine detail, 5-10 days for smoothing
# - **constant_sd:** Use global SD for stationary data, rolling SD for non-stationary
# - **separate_daytime_nighttime:** Use for data with strong diurnal cycles

print("\nParameter Selection Guide:")
print("  Sensitive: n_sd=2.0, winsize=48 (1-day)")
print("  Balanced: n_sd=2.5, winsize=96 (2-day)")
print("  Permissive: n_sd=3.5, winsize=240 (5-day)")
