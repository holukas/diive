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
print(
    f"  Day/night filtered: {filtered_dtnt.notna().sum()} ({100 * filtered_dtnt.notna().sum() / s_noise.notna().sum():.1f}%)")
print(
    f"  Global filtered: {filtered_global.notna().sum()} ({100 * filtered_global.notna().sum() / s_noise.notna().sum():.1f}%)")

# %%
# Parameter tuning: sensitivity analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Vary n_sigma threshold to understand its effect on outlier detection.
# Higher n_sigma → more permissive (fewer outliers), lower n_sigma → stricter.

print("\n" + "=" * 60)
print("Parameter Tuning: n_sigma Sensitivity")
print("=" * 60)

n_sigma_values = [3.0, 4.0, 5.5, 7.0, 9.0]

for n_sig in n_sigma_values:
    ham_tune = dv.Hampel(
        series=s_noise,
        n_sigma=n_sig,
        window_length=48 * 13,
        use_differencing=True,
        separate_day_night=False,
        showplot=False,
        verbose=0
    )
    ham_tune.calc(repeat=False)

    outlier_count = (ham_tune.flag == 2).sum()
    data_retained = ham_tune.filteredseries.notna().sum()
    outlier_rate = 100 * outlier_count / len(s_noise)

    print(f"\nn_sigma = {n_sig}:")
    print(f"  Outliers detected: {outlier_count} ({outlier_rate:.1f}%)")
    print(f"  Data retained: {data_retained} ({100 * data_retained / s_noise.notna().sum():.1f}%)")

# %%
# Parameter tuning: window length effect
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Shorter windows catch local anomalies, longer windows smooth regional trends.

print("\n" + "=" * 60)
print("Parameter Tuning: Window Length")
print("=" * 60)

window_lengths = [24 * 2, 48 * 5, 48 * 13, 48 * 30]  # 2 days to 30 days

for win_len in window_lengths:
    ham_win = dv.Hampel(
        series=s_noise,
        n_sigma=5.5,
        window_length=win_len,
        use_differencing=True,
        separate_day_night=False,
        showplot=False,
        verbose=0
    )
    ham_win.calc(repeat=False)

    outlier_count = (ham_win.flag == 2).sum()
    data_retained = ham_win.filteredseries.notna().sum()

    days = win_len / 48
    print(f"\nWindow = {win_len} steps ({days:.0f} days):")
    print(f"  Outliers detected: {outlier_count}")
    print(f"  Data retained: {data_retained}")

# %%
# Choosing parameters
# ^^^^^^^^^^^^^^^^^^^^
#
# **Guidelines for parameter selection:**
#
# - **n_sigma:** Start with 5.5 (typical), decrease (3-4) for stricter QC, increase (7+) if too aggressive
# - **window_length:** Use 10-15 days for typical QC, longer for smoothing, shorter for real-time detection
# - **use_differencing:** Enable for spike detection, disable for level-based thresholding
# - **separate_day_night:** Use when daytime/nighttime signals differ significantly

print("\nParameter Selection Summary:")
print("  Balanced (default): n_sigma=5.5, window=13 days")
print("  Strict QC: n_sigma=4.0, window=7 days")
print("  Permissive: n_sigma=7.0, window=20 days")
