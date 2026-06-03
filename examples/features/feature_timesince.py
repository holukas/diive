"""
===========================
Time Since Event Tracking
===========================

Demonstrates counting consecutive records since the last occurrence of a condition
using TimeSince class. Useful for dry period detection, frost period detection,
warm spell analysis, and event-based time tracking.

Best for: Tracking time intervals between occurrences of conditions.
"""

# %%
# Understanding TimeSince
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The TimeSince class counts the number of consecutive records since the last time
# a measurement fell outside specified limits. This is useful for tracking duration
# of conditions like dry periods, frost periods, or warm spells.
#
# Key parameter: `include_lim` controls whether the limit is inclusive or exclusive:
# - `include_lim=False` (exclusive): Records at exactly the limit value do NOT reset the counter
# - `include_lim=True` (inclusive): Records at exactly the limit value DO reset the counter
#
# Example: For precipitation with lower_lim=0 and include_lim=False:
# - Precipitation > 0 resets counter (counts as precipitation event)
# - Precipitation = 0 does NOT reset counter (exclusive, counts as dry)
# - This effectively counts dry periods (0 or near-0 precipitation records)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import diive as dv

# %%
# Dry Period Detection
# ^^^^^^^^^^^^^^^^^^^^
#
# Calculate time since last precipitation event.
# Demonstrates counting records since the last occurrence of precipitation > 0 mm,
# useful for identifying dry periods and drought conditions.
#
# Parameter explanation:
# - `lower_lim=0`: Only consider values > 0 as precipitation
# - `include_lim=False`: Make the limit exclusive, so precipitation = 0 does NOT reset counter

df = dv.load_exampledata_parquet()
series_prec = df.loc[(df.index.year == 2022) & (df.index.month == 7),
    "PREC_TOT_T1_25+20_1"].copy()

ts_prec = dv.variables.TimeSince(series_prec, lower_lim=0, include_lim=False)
ts_prec.calc()

print("Time Since Last Precipitation Event")
print("=" * 50)
print(
    f"Maximum dry period: {ts_prec.get_timesince().max()} records (~{ts_prec.get_timesince().max() * 0.5:.1f} hours)")
print(
    f"Mean dry period: {ts_prec.get_timesince().mean():.1f} records (~{ts_prec.get_timesince().mean() * 0.5:.1f} hours)")

# %%
# Understanding TimeSince Output Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The TimeSince class produces a DataFrame with these variables:
# - Original variable: The input measurement (e.g., PREC_TOT_T1_25+20_1)
# - UPPER_LIMIT: Upper limit from settings (if specified)
# - LOWER_LIMIT: Lower limit from settings
# - FLAG_IS_OUTSIDE_RANGE: Binary flag (1=outside limits, 0=inside limits)
# - TIMESINCE_*: Cumulative count of records since last occurrence outside limits

ts_prec_full_results = ts_prec.get_full_results()

print("\nFull results with all output variables:")
print(ts_prec_full_results.head(10))

print("\nOutput variables explained:")
print(f"  PREC_TOT_T1_25+20_1: Original measured precipitation (mm)")
print(f"  UPPER_LIMIT: {ts_prec_full_results['UPPER_LIMIT'].iloc[0]:.1f} (upper boundary from settings)")
print(f"  LOWER_LIMIT: {ts_prec_full_results['LOWER_LIMIT'].iloc[0]} (lower boundary from settings)")
print(f"  FLAG_IS_OUTSIDE_RANGE: 1 when value is outside limits (e.g., precipitation = 0)")
print(f"  TIMESINCE_PREC_TOT_T1_25+20_1: Number of records since last precipitation event")

# %%
# Direct Access to TimeSince Series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Two methods to access results:
# - `get_timesince()`: Returns only the TIMESINCE_* variable (faster, less memory)
# - `get_full_results()`: Returns all variables including flags and limits

ts_prec_series = ts_prec.get_timesince()

print("\nTimeSince series (direct access via get_timesince()):")
print(f"  Shape: {ts_prec_series.shape}")
print(f"  Data type: {ts_prec_series.dtype}")
print(f"  First 10 values:")
print(ts_prec_series.head(10).to_string())

# %%
# Visualize Dry Period Detection Time Series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ts_prec_df = ts_prec.get_full_results()

fig = plt.figure(facecolor='white', figsize=(16, 8), constrained_layout=True)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

dv.plotting.TimeSeries(series=ts_prec_df['PREC_TOT_T1_25+20_1']).plot(ax=ax1, color='#1565C0')
dv.plotting.TimeSeries(series=ts_prec_df['TIMESINCE_PREC_TOT_T1_25+20_1']).plot(ax=ax2, color='#D32F2F')
dv.plotting.TimeSeries(series=ts_prec_df['FLAG_IS_OUTSIDE_RANGE']).plot(ax=ax3, color='#00BCD4')

ax1.set_title("Measured Precipitation (mm)", fontsize=12, fontweight='bold')
ax2.set_title("Time Since Last Precipitation (records)", fontsize=12, fontweight='bold')
ax3.set_title("Flag (0=precipitation observed, 1=no precipitation)", fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=10)

for ax in [ax1, ax2, ax3]:
    ax.grid(True, alpha=0.3)

fig.show()

# %%
# Frost Period Detection
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate time since last freezing temperature event (temperature <= 0°C).
# Useful for identifying frost periods, freeze events, and cold spells.
#
# Parameter explanation:
# - `upper_lim=0`: Only consider values <= 0 as freezing
# - `include_lim=True`: Make the limit inclusive, so exactly 0°C counts as freezing temperature

df = dv.load_exampledata_parquet()
series_ta = df.loc[(df.index.year == 2022) & (df.index.month == 3),
    "Tair_f"].copy()

ts_ta = dv.variables.TimeSince(series_ta, upper_lim=0, include_lim=True)
ts_ta.calc()

print("\nTime Since Last Freezing Temperature")
print("=" * 50)
print(
    f"Maximum warm period: {ts_ta.get_timesince().max()} records (~{ts_ta.get_timesince().max() * 0.5:.1f} hours)")
print(
    f"Mean warm period: {ts_ta.get_timesince().mean():.1f} records (~{ts_ta.get_timesince().mean() * 0.5:.1f} hours)")
print(f"Temperature range: {series_ta.min():.1f}°C to {series_ta.max():.1f}°C")

# %%
# Interpreting Frost Period Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With upper_lim=0 and include_lim=True:
# - FLAG_IS_OUTSIDE_RANGE = 0 when temperature <= 0°C (freezing)
# - FLAG_IS_OUTSIDE_RANGE = 1 when temperature > 0°C (warm)
# - TIMESINCE_* counts records since temperature was last <= 0°C

ts_ta_full_results = ts_ta.get_full_results()

print("\nTemperature results (first 10 rows):")
print(ts_ta_full_results.head(10))

print("\nOutput variables explained:")
print(f"  Tair_f: Original measured air temperature (°C)")
print(f"  UPPER_LIMIT: {ts_ta_full_results['UPPER_LIMIT'].iloc[0]}")
print(f"  LOWER_LIMIT: {ts_ta_full_results['LOWER_LIMIT'].iloc[0]:.1f} (auto-calculated)")
print(f"  FLAG_IS_OUTSIDE_RANGE: 1 when temperature > 0°C (warm), 0 when temperature <= 0°C (freezing)")
print(f"  TIMESINCE_Tair_f: Records since last freezing temperature (<=0°C)")

# %%
# Time Series Visualization of Temperature Periods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ts_ta_df = ts_ta.get_full_results()

fig = plt.figure(facecolor='white', figsize=(16, 8), constrained_layout=True)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

dv.plotting.TimeSeries(series=ts_ta_df['Tair_f']).plot(ax=ax1, color='#1565C0')
dv.plotting.TimeSeries(series=ts_ta_df['TIMESINCE_Tair_f']).plot(ax=ax2, color='#D32F2F')
dv.plotting.TimeSeries(series=ts_ta_df['FLAG_IS_OUTSIDE_RANGE']).plot(ax=ax3, color='#00BCD4')

ax1.set_title("Measured Air Temperature (°C)", fontsize=12, fontweight='bold')
ax2.set_title("Time Since Last Freezing Temperature (records)", fontsize=12, fontweight='bold')
ax3.set_title("Flag (0=temperature <= 0°C, 1=temperature > 0°C)", fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=10)

for ax in [ax1, ax2, ax3]:
    ax.grid(True, alpha=0.3)

fig.show()

# %%
# Heatmap Visualization of Time-Since Patterns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize time-since patterns using heatmaps showing daily and hourly breakdown.
# Reveals which hours and days have the longest dry periods.

# Load example data - July 2022 precipitation
df = dv.load_exampledata_parquet()
series_prec = df.loc[(df.index.year == 2022) & (df.index.month == 7),
    "PREC_TOT_T1_25+20_1"].copy()

# Create TimeSince counter
ts_prec = dv.variables.TimeSince(series_prec, lower_lim=0, include_lim=False)
ts_prec.calc()
ts_prec_df = ts_prec.get_full_results()

# Create 3-panel heatmap comparison
fig = plt.figure(facecolor='white', figsize=(18, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

dv.plotting.HeatmapDateTime(series=ts_prec_df['PREC_TOT_T1_25+20_1']).plot(ax=ax1)
dv.plotting.HeatmapDateTime(series=ts_prec_df['TIMESINCE_PREC_TOT_T1_25+20_1']).plot(ax=ax2)
dv.plotting.HeatmapDateTime(series=ts_prec_df['FLAG_IS_OUTSIDE_RANGE']).plot(ax=ax3)

ax1.set_title("Precipitation (mm)", fontsize=12, fontweight='bold')
ax2.set_title("Time Since Last Precipitation (records)", fontsize=12, fontweight='bold')
ax3.set_title("Flag (0=precipitation, 1=dry)", fontsize=12, fontweight='bold')

# Remove left y-axis labels for middle and right plots
ax2.tick_params(left=True, labelleft=False)
ax3.tick_params(left=True, labelleft=False)

fig.show()
