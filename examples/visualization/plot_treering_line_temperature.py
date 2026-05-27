"""
==========================================
Tree-Ring Air Temperature Line Plot
==========================================

Circular spiral plot showing daily air temperature as concentric annual line traces.

Each ring represents one year (inner = older, outer = more recent). The line for
each year traces one full revolution around the circle — January at the bottom,
July at the top. Radial displacement from the ring baseline encodes the temperature:
values above the global midpoint push outward, values below pull inward.
"""

# %%
# Load data
# ^^^^^^^^^^
#
# Load the 30-minute example data directly.  The class handles resampling
# internally via the ``resample_freq`` parameter, so no manual pre-processing
# is needed.

import matplotlib.pyplot as plt

import diive as dv

df = dv.load_exampledata_parquet()

print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} half-hourly records")
print(f"Tair_f range: {df['Tair_f'].min():.1f} to {df['Tair_f'].max():.1f} deg C")

# %%
# Create tree-ring object
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Data is prepared once in ``__init__``; both ``plot()`` and ``plot_line()`` reuse
# the same grid without recomputation.

tr = dv.plotting.TreeRingPlot(
    df=df,
    value_col='Tair_f',  # Data: column with values to visualize
    resample_freq='D'  # Data: resample to daily means (366 slots per ring)
)

# %%
# Radial line plot — single color
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each year traces one full revolution as a line.  ``amplitude_scale=0.45``
# (default) keeps lines within their ring boundaries.  ``vmin``/``vmax`` set
# the normalization anchor: the global midpoint (-20+20)/2 = 0 deg C maps to
# the baseline, so the line radiates outward for warm periods and inward for cold.

tr.plot_line(
    figsize=(10, 10),
    title='Air temperature at ICOS Ecosystem Station Davos (2013-2022)',
    vmin=-20,
    vmax=20,
    color='#2196F3',  # Styling: Material blue for all years
    linewidth=0.8,
    alpha=0.85,
    amplitude_scale=0.45,  # Styling: line spans up to 90% of ring width
    show_month_labels=True,
    show_year_labels=True,
    show_year_separators=True,
    year_label_frequency=1,
)

plt.show(block=False)

# %%
# Radial line plot — colored by year with fill
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Setting ``cmap`` assigns a color to each year along the colormap, from oldest
# (low end) to most recent (high end).  ``fill=True`` shades the area between
# the baseline and the line, making above/below-midpoint regions immediately
# visible.

tr.plot_line(
    figsize=(10, 10),
    title='Air temperature at ICOS Ecosystem Station Davos (2013-2022)',
    vmin=-20,
    vmax=20,
    cmap='plasma',  # Styling: oldest years dark, newest years bright
    linewidth=0.9,
    alpha=0.9,
    amplitude_scale=0.45,
    fill=True,  # Styling: shade between baseline and line
    fill_alpha=0.15,
    show_month_labels=True,
    show_year_labels=True,
    show_year_separators=True,
    year_label_frequency=1,
)

plt.show(block=False)

print("Plotted radial line tree-ring air temperature visualization")
