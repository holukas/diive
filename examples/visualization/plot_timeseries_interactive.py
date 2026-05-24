"""
====================================
Interactive Time Series with Bokeh
====================================

Create interactive time series plots using Bokeh.
Provides zoom, pan, selection, and export tools for exploring data.

Best for: Interactive data exploration, dynamic visualization, web-based dashboards
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
# Basic interactive plot
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Create an interactive Bokeh plot with default size.
# Hover over points to see tooltips with date and value.
# Use the toolbar for zoom, pan, selection, and data export.

series = df['NEE_CUT_REF_f'].copy()

ts = dv.plotting.TimeSeries(
    series=series
)
ts.plot_interactive(
    height=600,  # Plot height in pixels
    width=1200   # Plot width in pixels
)

print("\nCreated interactive NEE flux time series")
print("Tools available: Hover (tooltip), Box Zoom, Reset, Pan, Box Select, Wheel Zoom, Undo, Redo, Save")

# %%
# Larger interactive plot for detailed exploration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a larger interactive plot for detailed examination of patterns.

series_temp = df['Tair_f'].copy()

ts_large = dv.plotting.TimeSeries(
    series=series_temp
)
ts_large.plot_interactive(
    height=800,  # Taller plot
    width=1600   # Wider plot
)

print("\nCreated larger interactive temperature plot")
print("Larger size allows for more detailed pattern exploration and visual inspection")
