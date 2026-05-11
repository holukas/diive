"""
=======================
Anomaly Visualization
=======================

Long-term anomalies per year compared to a reference period.
Shows deviations from climatological mean as bar plots (red=above reference, blue=below).
Reference period mean ± standard deviation provided for context.

Best for: Detecting long-term trends, identifying extreme years, climate analysis
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# Calculate annual air temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Aggregate daily data to annual means for anomaly calculation.
# Anomalies show deviations from a reference period baseline.

series = df['Tair_f'].copy()
series = series.resample('YE').mean()
series.index = series.index.year

series_label = "Air temperature"

# %%
# Temperature anomalies with reference period
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize yearly anomalies relative to a reference period (2015-2017).
# Red bars indicate years warmer than the reference mean, blue bars indicate cooler years.
# Reference statistics and last 10 years summary displayed in the plot.

anomaly_plot = dv.plot_longterm_anomalies_year(
    series=series,                    # Annual mean temperature per year
    series_label=series_label,        # Variable name for plot title
    series_units='(°C)',              # Units appended to y-axis label
    reference_start_year=2015,        # First year of reference period
    reference_end_year=2017           # Last year of reference period
)
anomaly_plot.plot(
    ax=None,                          # Create new figure
    title='Annual Air Temperature Anomalies (2013-2022)'  # Custom title
)

print("\nPlotted air temperature anomalies")
print(f"Reference period: 2015-2017 ({2017 - 2015 + 1} years)")

# %%
# Statistics
# ^^^^^^^^^^

print(f"\nTemperature statistics:")
print(f"  Annual mean: {series.mean():.2f}°C")
print(f"  Annual std dev: {series.std():.2f}°C")
print(f"  Warmest year: {series.idxmax()} ({series.max():.2f}°C)")
print(f"  Coolest year: {series.idxmin()} ({series.min():.2f}°C)")
print(f"  Range: {series.max() - series.min():.2f}°C")
