"""
============================
HeatmapDateTime (Basic)
============================

Temporal heatmaps for time series visualization: date × time-of-day grids.

Best for: Visualizing daily patterns, identifying diurnal cycles
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# HeatmapDateTime - vertical orientation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Dates on y-axis, time-of-day (0-24h) on x-axis.

series = df['NEE_CUT_REF_f'].copy()
series = series.loc[series.index.year >= 2020]
series = series.dropna()

hm = dv.plot_heatmap_datetime(
    series=series,
    title="NEE flux (vertical orientation)",
    vmin=-10,                          # Minimum color value
    vmax=10,                           # Maximum color value
    ax_orientation="vertical",         # Dates on y-axis
    show_values=False,                 # Don't show values on cells
    cmap='RdBu_r'                      # Colormap
)
hm.show()

print("Plotted HeatmapDateTime in vertical orientation")

# %%
# HeatmapDateTime - horizontal orientation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Time-of-day on y-axis, dates on x-axis.

hm = dv.plot_heatmap_datetime(
    series=series,
    title="NEE flux (horizontal orientation)",
    vmin=-10,                          # Minimum color value
    vmax=10,                           # Maximum color value
    ax_orientation="horizontal",       # Time on y-axis
    show_values=False,                 # Don't show values on cells
    cmap='RdBu_r'                      # Colormap
)
hm.show()

print("Plotted HeatmapDateTime in horizontal orientation")
