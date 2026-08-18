"""
===================
Potential Radiation
===================

Calculate potential shortwave incoming radiation with ``dv.variables.potrad``,
a faithful port of ONEFlux's ``get_rpot`` -- the routine that produces the
SW_IN_POT column of FLUXNET/AmeriFlux/ICOS products.

Potential radiation is the shortwave radiation arriving on a horizontal surface
at the top of the atmosphere above the site: no atmospheric attenuation, no
clouds. It follows from solar geometry alone, so it needs nothing but a
timestamp index and the site coordinates: no gaps, no sensor, available for any
period. That makes it the reference for day/night classification, gap-filling,
timestamp-shift detection, and radiation quality checks.

Best for: a top-of-atmosphere radiation reference and solar geometry at a site.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/feature_potentialradiation.png'

# %%
# Calculate potential radiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``potrad`` takes a timestamp index and the site coordinates. The averaging
# period is inferred from the spacing of the index (30 min here), and timestamps
# are read as TIMESTAMP_MIDDLE, the diive convention: the record at ``t`` covers
# ``[t - 15min, t + 15min)``.
#
# ``utc_offset`` is the local standard time offset. Daylight saving is never
# applied, following FLUXNET convention.

import calendar

import matplotlib.pyplot as plt
import pandas as pd

import diive as dv

LAT = 47.286417  # Davos, decimal degrees north
LON = 7.733750  # decimal degrees east
UTC_OFFSET = 1  # local standard time, CET (UTC+1)

# Load example data, single year
df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2018].copy()

sw_in_pot = dv.variables.potrad(
    timestamp_index=df.index,
    lat=LAT,
    lon=LON,
    utc_offset=UTC_OFFSET
)

print("Potential radiation")
print("=" * 50)
print(f"Period : {sw_in_pot.index[0]} to {sw_in_pot.index[-1]}")
print(f"Name   : {sw_in_pot.name}")
print(f"Maximum: {sw_in_pot.max():.1f} W/m²")
print(f"Mean   : {sw_in_pot.mean():.1f} W/m²")

# %%
# What the port computes
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Four properties carry the FLUXNET parity:
#
# - Solar constant 1376 W/m², ONEFlux's own value.
# - Spencer (1971) series for solar declination and for the earth-sun distance,
#   so the annual eccentricity cycle of about +/-3.4% in irradiance is included.
# - True solar noon from the NOAA solar position algorithm, i.e. the equation of
#   time is part of the calculation (next section).
# - The returned value is the mean over each averaging period, computed from
#   1-minute steps, not an instantaneous value at the timestamp (section after
#   next).
#
# Declination sets the seasonal envelope: the noon sun stands highest around the
# June solstice and lowest around the December one, which at this latitude
# changes the daily peak by more than a factor of two.

peak_by_month = sw_in_pot.groupby(sw_in_pot.index.month).max()

print("\nHighest potential radiation per month")
print("=" * 50)
for month, peak in peak_by_month.items():
    print(f"  {calendar.month_abbr[month]}: {peak:7.1f} W/m²")

# %%
# Solar noon follows the equation of time
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The daily curve is centred on true solar noon, not on 12:00 local clock time.
# Two effects separate the two:
#
# - Longitude. The site is at 7.73 degrees east while CET is referenced to
#   15 degrees east, which puts solar noon roughly half an hour late.
# - The equation of time. Earth's elliptical orbit and axial tilt shift solar
#   noon by up to about a quarter of an hour either way over the year.
#
# The 30-minute record is too coarse to resolve that swing, so compute on a
# 2-minute index and take the time of each day's peak. ``potrad`` infers the
# 2-minute period from the index; nothing else changes.

fine_index = pd.date_range('2018-01-01 00:01', '2018-12-31 23:59', freq='2min')
sw_in_pot_fine = dv.variables.potrad(
    timestamp_index=fine_index,
    lat=LAT,
    lon=LON,
    utc_offset=UTC_OFFSET
)

# Time of day of each day's peak, in minutes since midnight
daily_peak = sw_in_pot_fine.groupby(sw_in_pot_fine.index.date).idxmax()
peak_minutes = pd.Series(
    [t.hour * 60 + t.minute for t in daily_peak],
    index=pd.DatetimeIndex(daily_peak.index)
)
monthly_noon = peak_minutes.groupby(peak_minutes.index.month).mean()


def _as_clocktime(minutes: float) -> str:
    """Format minutes since midnight as HH:MM."""
    return f"{int(minutes // 60):02d}:{int(round(minutes % 60)):02d}"


print("\nMean time of the daily peak, by month (local standard time)")
print("=" * 50)
for month, minutes in monthly_noon.items():
    print(f"  {calendar.month_abbr[month]}: {_as_clocktime(minutes)}")

print(f"\nEarliest peak in 2018: {_as_clocktime(peak_minutes.min())}")
print(f"Latest peak in 2018  : {_as_clocktime(peak_minutes.max())}")
print(f"Swing                : {peak_minutes.max() - peak_minutes.min():.0f} minutes")
print("A reference that pins solar noon to a fixed clock time misses this swing,")
print("which is what makes it unusable for detecting timestamp shifts.")

# %%
# The value is a period mean
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ONEFlux resolves potential radiation on a 1-minute grid and averages it over
# the averaging period. The 30-minute record at ``t`` is therefore the mean of
# the minutes in ``[t - 15min, t + 15min)``, not the value at ``t``. This is why
# a record spanning sunrise is small but non-zero instead of jumping from 0 to
# the full value.
#
# The averaging also makes the output consistent under resampling: an hourly
# record equals the mean of the two 30-minute records inside it. Below, both are
# computed independently and compared.

# TIMESTAMP_MIDDLE: 30-minute records are centred at :15 and :45, hourly at :30
idx_30min = pd.date_range('2018-06-21 00:15', '2018-06-21 23:45', freq='30min')
idx_hourly = pd.date_range('2018-06-21 00:30', '2018-06-21 23:30', freq='60min')

pot_30min = dv.variables.potrad(timestamp_index=idx_30min, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
pot_hourly = dv.variables.potrad(timestamp_index=idx_hourly, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)

# Average each pair of 30-minute records onto the hourly middle timestamp (:30)
pot_30min_paired = pot_30min.groupby(pot_30min.index.floor('60min') + pd.Timedelta(minutes=30)).mean()
comparison = pd.DataFrame({'hourly': pot_hourly, 'mean_of_two_30min': pot_30min_paired})

print("\nPeriod mean, sunrise on 21 June 2018")
print("=" * 50)
print(comparison.loc['2018-06-21 03:30':'2018-06-21 07:30'].round(2).to_string())

max_diff = (comparison['hourly'] - comparison['mean_of_two_30min']).abs().max()
print(f"\nLargest difference over the day: {max_diff:.2e} W/m²")

# %%
# Diurnal curves
# ^^^^^^^^^^^^^^
#
# Left: the potential radiation curve on three days of 2018. Right: measured
# radiation during a summer week, drawn against the potential envelope. Measured
# values stay under the envelope; how far under is the cloud signal that quality
# checks look for.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor='white', constrained_layout=True)

selected_days = {
    '2018-06-21': ('Summer solstice (21 Jun)', '#F4511E'),
    '2018-03-20': ('Equinox (20 Mar)', '#43A047'),
    '2018-12-21': ('Winter solstice (21 Dec)', '#1E88E5'),
}

for date, (label, color) in selected_days.items():
    day = sw_in_pot.loc[date]
    hour_of_day = day.index.hour + day.index.minute / 60
    ax1.plot(hour_of_day, day.values, color=color, linewidth=2, label=label)
    ax1.fill_between(hour_of_day, 0, day.values, color=color, alpha=0.15)

ax1.axvline(12, color='#455A64', linestyle=':', linewidth=1.2, label='12:00 local clock time')
ax1.set_title('Potential radiation over the day', fontsize=11, fontweight='bold')
ax1.set_xlabel('Hour of day (local standard time)', fontsize=9)
ax1.set_ylabel('SW_IN_POT (W/m²)', fontsize=9)
ax1.set_xlim(0, 24)
ax1.set_xticks(range(0, 25, 3))
ax1.tick_params(labelsize=8)
ax1.legend(loc='upper left', fontsize=8, frameon=True)
ax1.grid(True, alpha=0.3)

week = slice('2018-07-08', '2018-07-14')
ax2.fill_between(sw_in_pot.loc[week].index, 0, sw_in_pot.loc[week].values,
                 color='#455A64', alpha=0.2, label='SW_IN_POT (potential)')
ax2.plot(sw_in_pot.loc[week].index, sw_in_pot.loc[week].values, color='#455A64', linewidth=1.2)
ax2.plot(df.loc[week].index, df.loc[week, 'Rg_f'], color='#F4511E', linewidth=1.2, label='Rg_f (measured)')
ax2.set_title('Measured radiation against the potential envelope', fontsize=11, fontweight='bold')
ax2.set_xlabel('Date', fontsize=9)
ax2.set_ylabel('Radiation (W/m²)', fontsize=9)
ax2.tick_params(labelsize=8)
ax2.tick_params(axis='x', rotation=30)
ax2.legend(loc='upper left', fontsize=8, frameon=True)
ax2.grid(True, alpha=0.3)

fig.show()

# %%
# Annual pattern
# ^^^^^^^^^^^^^^
#
# The heatmap shows every 30-minute record of 2018: calendar day along one axis,
# time of day along the other. Day length and the seasonal cycle of the peak are
# both visible in one view.

dv.plotting.HeatmapDateTime(
    series=sw_in_pot,
    ax_orientation='horizontal'
).plot(
    format_style=dv.plotting.FormatStyle(title='Potential Shortwave Radiation - Daily & Hourly Patterns'),
    zlabel='W/m²',
    cb_digits_after_comma=0,
    figsize=(14, 6)
)
