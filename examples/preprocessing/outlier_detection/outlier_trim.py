"""
==============
Trim Filter
==============

Symmetric trimming approach: remove values below threshold,
then remove equal number of values from high end (trimmed mean).
"""

# %%
# Create test data with synthetic noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate temperature data and add impulse noise.

import diive as dv

df = dv.load_exampledata_parquet()

s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()

# Add synthetic impulse noise
s_noise = dv.variables.add_impulse_noise(
    series=s,
    factor_low=-10,
    factor_high=4,
    contamination=0.04,
    seed=42
)
s_noise.name = f"{s.name}+noise"

print("Test data created:")
print(f"  Records: {len(s_noise)}")
print(f"  Valid: {s_noise.notna().sum()}")
print(f"  Range: {s_noise.min():.2f} to {s_noise.max():.2f}°C")

# %%
# Trim nighttime only
# ^^^^^^^^^^^^^^^^^^^
#
# Apply trimming to nighttime data only.
# Useful when daytime varies widely but nighttime is stable.

trim_night = dv.outliers.TrimLow(
    series=s_noise,
    trim_daytime=False,
    trim_nighttime=True,
    lower_limit=-75,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1
)
trim_night.calc()

flag_night = trim_night.overall_flag
filtered_night = s_noise.copy()
filtered_night.loc[flag_night == 2] = None

print("\nNighttime trimming results:")
print(f"  Outliers detected: {(flag_night == 2).sum()}")
print(f"  Valid records: {filtered_night.notna().sum()}")
print(f"  Data retained: {100*filtered_night.notna().sum()/s_noise.notna().sum():.1f}%")

# %%
# Trim daytime only
# ^^^^^^^^^^^^^^^^^
#
# Apply trimming to daytime data only.
# Useful when daytime has measurement issues but nighttime is reliable.

trim_day = dv.outliers.TrimLow(
    series=s_noise,
    trim_daytime=True,
    trim_nighttime=False,
    lower_limit=-75,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1
)
trim_day.calc()

flag_day = trim_day.overall_flag
filtered_day = s_noise.copy()
filtered_day.loc[flag_day == 2] = None

print("\nDaytime trimming results:")
print(f"  Outliers detected: {(flag_day == 2).sum()}")
print(f"  Valid records: {filtered_day.notna().sum()}")
print(f"  Data retained: {100*filtered_day.notna().sum()/s_noise.notna().sum():.1f}%")

# %%
# Trim daytime and nighttime together
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With both enabled, each period is trimmed independently against its own
# distribution and the flags are combined. Use this when the plausible range
# differs between day and night and both periods need screening.

trim_both = dv.outliers.TrimLow(
    series=s_noise,
    trim_daytime=True,
    trim_nighttime=True,
    lower_limit=-75,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1
)
trim_both.calc()

flag_both = trim_both.overall_flag
filtered_both = s_noise.copy()
filtered_both.loc[flag_both == 2] = None

print("\nDaytime + nighttime trimming results:")
print(f"  Outliers detected: {(flag_both == 2).sum()}")
print(f"  Valid records: {filtered_both.notna().sum()}")
print(f"  Data retained: {100*filtered_both.notna().sum()/s_noise.notna().sum():.1f}%")

# %%
# Symmetric tails: it also removes high values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For every value below ``lower_limit``, an equal number of the highest values
# is rejected too, keeping the trim symmetric (the trimmed-mean rationale). So
# some rejected records sit above the limit, not below it. Use a one-sided
# method (e.g. AbsoluteLimits) if you only want to drop the low extremes.

rejected_both = s_noise[flag_both == 2]
print("\nWhat the both-modes trim rejected:")
print(f"  Below lower_limit (-75): {(rejected_both < -75).sum()}")
print(f"  At/above lower_limit (high tail): {(rejected_both >= -75).sum()}")

# %%
# Comparison
# ^^^^^^^^^^
#
# Trimming removes symmetric tails from the distribution.
# Daytime and nighttime filtering can be combined or applied selectively.

print("\nComparison:")
print(f"Original valid: {s_noise.notna().sum()}")
print(f"Nighttime trim: {filtered_night.notna().sum()} retained")
print(f"Daytime trim: {filtered_day.notna().sum()} retained")
print(f"Day+night trim: {filtered_both.notna().sum()} retained")
