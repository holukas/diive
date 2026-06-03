"""
============================
Time Series Visualization
============================

Interactive time series plots for exploring temporal patterns in data.
Shows multiple customization options for time series line plots.

Best for: Visualizing temporal data with axis control, unit labels, and colors
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
# Filter to one month for clearer visualization
df = df.loc[(df.index.year == 2022) & (df.index.month == 7)].copy()
print(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")

# %%
# Basic time series plot
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Create a simple time series plot with default styling.

series = df['NEE_CUT_REF_f'].copy()

ts = dv.plotting.TimeSeries(
    series=series
)
ts.plot(
    ax=None,  # Create new figure
    color=None,  # Use default theme color
    series_units=None,  # No units label
    xlabel=None,  # Use default 'Date'
    ylabel=None,  # Use series name
    title=None  # Use series name
)

print("\nPlotted NEE flux time series")

# %%
# Time series with custom units and labels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add units information and custom labels for better documentation.

ts_with_units = dv.plotting.TimeSeries(
    series=series
)
ts_with_units.plot(
    ax=None,
    color=None,
    series_units=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',  # Add units to y-axis
    xlabel='Date',  # Custom x-axis label
    ylabel='CO₂ Flux',  # Custom y-axis label
    title='Net Ecosystem Exchange (2013-2022)'  # Custom title
)

print("\nPlotted with custom units, labels, and title")

# %%
# Time series with custom color
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Customize line color for publication-quality plots.

series_temp = df['Tair_f'].copy()

ts_color = dv.plotting.TimeSeries(
    series=series_temp
)
ts_color.plot(
    ax=None,
    color='#FF6B6B',  # Custom line color (red)
    series_units='(°C)',  # Temperature units
    xlabel='Date',
    ylabel='Air Temperature',
    title='Air Temperature Time Series'
)

print("\nPlotted temperature with custom color")

# %%
# Gaps stay visible, with point markers
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# By default missing values are kept, so the line breaks at gaps instead of
# drawing a continuous line across missing periods (which would misrepresent
# data coverage). Use the measured (un-gap-filled) flux to see this, and enable
# point markers. Pass ``drop_gaps=True`` if you instead want NaNs removed.

series_measured = df['NEE_CUT_REF_orig'].copy()  # measured NEE, contains gaps

ts_gaps = dv.plotting.TimeSeries(
    series=series_measured
)
ts_gaps.plot(
    ax=None,
    series_units=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    ylabel='Measured NEE',
    title='Measured NEE — gaps shown as line breaks',
    marker=True,       # draw a marker at each observation
    linewidth=1.2,     # thinner line
    alpha=0.85
)

print("\nPlotted measured NEE with visible gaps and point markers")
