"""
=======================
Anomaly Visualization
=======================

Long-term anomalies per year compared to a reference period.
Shows deviations from climatological mean as bar plots.

Best for: Detecting long-term trends, identifying extreme years, climate analysis
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# Temperature anomalies by year
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show yearly air temperature anomalies (red=above reference, blue=below).
# Reference period: 2015-2017.

series = df['Tair_f'].copy()
series = series.resample('YE').mean()
series.index = series.index.year

series_label = "CH-DAV: Air temperature"

dv.plot_longterm_anomalies_year(
    series=series,
    series_label=series_label,
    series_units='(°C)',
    reference_start_year=2015,
    reference_end_year=2017
).plot()

print("\nPlotted air temperature anomalies (reference: 2015-2017)")

# %%
# Statistics
# ^^^^^^^^^^

print(f"\nTemperature statistics:")
print(f"  Annual mean: {series.mean():.2f}°C")
print(f"  Annual std dev: {series.std():.2f}°C")
print(f"  Warmest year: {series.idxmax()} ({series.max():.2f}°C)")
print(f"  Coolest year: {series.idxmin()} ({series.min():.2f}°C)")
