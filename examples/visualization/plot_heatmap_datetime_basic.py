"""
============================
HeatmapDateTime (Basic)
============================

Temporal heatmaps for time series visualization: date × time-of-day grids.
Shows daily patterns with optional custom styling.

Best for: Visualizing daily patterns, identifying diurnal cycles
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
series = df['NEE_CUT_REF_f'].copy()
series = series.loc[series.index.year >= 2020]
series = series.dropna()

print(f"Loaded {len(series)} records from {series.index[0].date()} to {series.index[-1].date()}")

# %%
# A shared FormatStyle for both orientations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# FormatStyle carries the plot *chrome* (here just the title) so it can be built
# once and reused across both heatmaps. The colorbar settings (vmin/vmax/cmap/
# zlabel) are NOT chrome - they stay direct plot() arguments because FormatStyle
# does not own the colorbar. We override only the title per call via .merged().

style = dv.plotting.FormatStyle(title="NEE flux")

# %%
# HeatmapDateTime - vertical orientation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Dates on y-axis, time-of-day (0-24h) on x-axis.
# Vertical layout is intuitive: time progresses left-to-right, days progress top-to-bottom.

hm = dv.plotting.HeatmapDateTime(
    series=series,
    ax_orientation="vertical"  # Dates on y-axis, hours on x-axis
)
hm.plot(
    ax=None,  # Create new figure
    format_style=style.merged(title="NEE flux - Vertical (dates on y-axis)"),
    vmin=-10,  # Minimum color value
    vmax=10,  # Maximum color value
    cmap='RdBu_r',  # Colormap
    zlabel=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    show_values=False  # Don't overlay cell values
)

print("Plotted HeatmapDateTime in vertical orientation")

# %%
# HeatmapDateTime - horizontal orientation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Time-of-day on y-axis, dates on x-axis.
# Horizontal layout places dates along x-axis for easier date reading.
# The same FormatStyle is reused, again with just the title overridden.

hm = dv.plotting.HeatmapDateTime(
    series=series,
    ax_orientation="horizontal"  # Dates on x-axis, hours on y-axis
)
hm.plot(
    ax=None,  # Create new figure
    format_style=style.merged(title="NEE flux - Horizontal (dates on x-axis)"),
    vmin=-10,  # Minimum color value
    vmax=10,  # Maximum color value
    cmap='RdBu_r',  # Colormap
    zlabel=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    show_values=False  # Don't overlay cell values
)

print("Plotted HeatmapDateTime in horizontal orientation")
