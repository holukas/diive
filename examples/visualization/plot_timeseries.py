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
# Create a simple time series plot with default styling. Passing no chrome
# arguments at all keeps the diive house style: 'Date' on the x-axis, the
# series name as title and y-label.

series = df['NEE_CUT_REF_f'].copy()

ts = dv.plotting.TimeSeries(
    series=series
)
ts.plot(ax=None)  # New figure, default chrome

print("\nPlotted NEE flux time series")

# %%
# Time series with custom units and labels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Chrome (title, axis labels, units, font sizes, ...) is described once in a
# ``FormatStyle`` and handed to ``plot()`` via ``format_style=``. Data-rendering
# arguments like ``color`` stay direct on ``plot()``.

ts_with_units = dv.plotting.TimeSeries(
    series=series
)
ts_with_units.plot(
    ax=None,
    color=None,  # Use default theme color
    format_style=dv.plotting.FormatStyle(
        title='Net Ecosystem Exchange (2013-2022)',
        xlabel='Date',
        ylabel='CO₂ Flux',
        yunits=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    ),
)

print("\nPlotted with custom units, labels, and title")

# %%
# Time series with custom color
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``color`` is a data-rendering choice, so it stays a direct ``plot()`` argument
# while the chrome lives in the ``FormatStyle``.

series_temp = df['Tair_f'].copy()

ts_color = dv.plotting.TimeSeries(
    series=series_temp
)
ts_color.plot(
    ax=None,
    color='#FF6B6B',  # Custom line color (red)
    format_style=dv.plotting.FormatStyle(
        title='Air Temperature Time Series',
        xlabel='Date',
        ylabel='Air Temperature',
        yunits='(°C)',
    ),
)

print("\nPlotted temperature with custom color")

# %%
# Reuse one FormatStyle across plots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The key benefit of the shared style: build it once and hand the same instance
# to several plots so they look identical. Here one style formats both the
# gap-filled and the measured NEE series.

shared_style = dv.plotting.FormatStyle(
    xlabel='Date',
    ylabel='NEE',
    yunits=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
)

dv.plotting.TimeSeries(series=series).plot(
    ax=None,
    format_style=shared_style,
)
dv.plotting.TimeSeries(series=df['NEE_CUT_REF_orig'].copy()).plot(
    ax=None,
    format_style=shared_style,
)

print("\nPlotted two series sharing one FormatStyle")

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
    marker=True,       # draw a marker at each observation
    linewidth=1.2,     # thinner line
    alpha=0.85,
    format_style=dv.plotting.FormatStyle(
        title='Measured NEE — gaps shown as line breaks',
        ylabel='Measured NEE',
        yunits=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    ),
)

print("\nPlotted measured NEE with visible gaps and point markers")
