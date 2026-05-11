"""
============================
Time Series Visualization
============================

Interactive time series plots using matplotlib and Bokeh.
Demonstrates basic and customized time series visualization options.

Best for: Visualizing temporal data with interactive zoom and pan controls
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

print(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")

# %%
# Basic interactive time series plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a simple time series plot with default settings.

series = df['NEE_CUT_REF_f'].copy()

ts = dv.plot_time_series(series=series)
ts.plot()

print("\nPlotted NEE flux time series")

# %%
# Time series with custom units
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add units information to the plot for better documentation.

ts_with_units = dv.plot_time_series(
    series=series,
    series_units=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
)
ts_with_units.plot()

print("\nPlotted with custom units label")

# %%
# Multiple series on same plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare two variables side by side.

series_temp = df['Tair_f'].copy()

print(f"\nNEE range: {series.min():.1f} to {series.max():.1f}")
print(f"Temperature range: {series_temp.min():.1f} to {series_temp.max():.1f}°C")
