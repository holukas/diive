"""
=======================
Local Outlier Factor
=======================

Density-based outlier detection identifying values with anomalous local neighborhoods.
Effective for detecting contextual outliers where local density varies.
"""

# %%
# Create test data with synthetic noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate summer temperature data and add impulse noise to simulate spikes.

import diive as dv

df = dv.load_exampledata_parquet()
s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()
s = s.loc[s.index.month == 7].copy()

s_noise = dv.variables.add_impulse_noise(
    series=s,
    factor_low=-10,
    factor_high=3,
    contamination=0.04
)
s_noise.name = f"{s.name}+noise"

print("Test data created:")
print(f"  Records: {len(s_noise)}")
print(f"  Valid: {s_noise.notna().sum()}")
print(f"  Range: {s_noise.min():.2f} to {s_noise.max():.2f}°C")

# %%
# LOF with day/night separation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use separate contamination rates for daytime and nighttime.
# LOF identifies points with anomalous local density compared to neighbors.

lof_dtnt = dv.outliers.LocalOutlierFactor(
    series=s_noise,
    n_neighbors=20,
    contamination=0.05,
    separate_daytime_nighttime=True,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1,
    n_jobs=-1
)

lof_dtnt.calc(repeat=False)

flag_dtnt = lof_dtnt.get_flag()
filtered_dtnt = s_noise.copy()
filtered_dtnt.loc[flag_dtnt == 2] = None

print("\nDay/night LOF results:")
print(f"  Outliers detected: {(flag_dtnt == 2).sum()}")
print(f"  Valid records: {filtered_dtnt.notna().sum()}")
print(f"  Data retained: {100*filtered_dtnt.notna().sum()/s_noise.notna().sum():.1f}%")

# %%
# LOF with global threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Single contamination rate applied to entire series.
# Simpler approach when time-of-day variation is not significant.

lof_global = dv.outliers.LocalOutlierFactor(
    series=s_noise,
    n_neighbors=20,
    contamination=0.05,
    separate_daytime_nighttime=False,
    showplot=False,
    verbose=1,
    n_jobs=-1
)

lof_global.calc(repeat=False)

flag_global = lof_global.get_flag()
filtered_global = s_noise.copy()
filtered_global.loc[flag_global == 2] = None

print("\nGlobal LOF results:")
print(f"  Outliers detected: {(flag_global == 2).sum()}")
print(f"  Valid records: {filtered_global.notna().sum()}")
print(f"  Data retained: {100*filtered_global.notna().sum()/s_noise.notna().sum():.1f}%")

# %%
# Comparison
# ^^^^^^^^^^
#
# LOF detects density-based anomalies effectively for both approaches.

print("\nComparison:")
print(f"Original valid: {s_noise.notna().sum()}")
print(f"Day/night LOF: {filtered_dtnt.notna().sum()} retained")
print(f"Global LOF: {filtered_global.notna().sum()} retained")
