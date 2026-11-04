"""
==========================================
Tree-Ring Air Temperature Line Plot
==========================================

Radial line plot of daily air temperature. Each year is one line around a full circle.

The oldest year is at the center and the newest at the outer edge. Within its ring,
each line moves outward for higher values and inward for lower values. The line color
shows the same value: low temperatures are blue, high temperatures are red. With a
small ``ring_width`` the lines of neighbouring years overlap.
"""

# %%
# Load data
# ^^^^^^^^^^

import matplotlib.pyplot as plt

import diive as dv

df = dv.load_exampledata_parquet()

print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} half-hourly records")
print(f"Tair_f range: {df['Tair_f'].min():.1f} to {df['Tair_f'].max():.1f} deg C")

# %%
# Create tree-ring object
# ^^^^^^^^^^^^^^^^^^^^^^^^

tr = dv.plotting.TreeRingPlot(
    df=df,
    value_col='Tair_f',
    resample_freq='D'
)

# %%
# Radial line plot — default blue-to-red colormap
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# The shared FormatStyle carries the title (chrome). This is a polar plot, so
# only the title and its font fields apply; the colorbar (cb_*) stays direct.
style = dv.plotting.FormatStyle(
    title='Air temperature at ICOS Ecosystem Station Davos (2013-2022)',
)

tr.plot_line(
    figsize=(10, 10),
    format_style=style,
    # vmin=-15,
    # vmax=25,
    linewidth=1.2,
    alpha=1,
    amplitude_scale=1,
    ring_width=0.04,
    show_month_labels=True,
    show_year_labels=False,
    cb_label='Air temperature (deg C)',
)

plt.show(block=False)

print("Plotted radial line tree-ring air temperature visualization")
