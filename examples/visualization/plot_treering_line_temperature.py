"""
==========================================
Tree-Ring Air Temperature Line Plot
==========================================

Circular spiral plot showing daily air temperature as overlapping annual line traces.

Both the radial position and the line color encode the same data value: low temperatures
are blue and near the center, high temperatures are red and near the outer edge.
All years share the same y-axis so lines overlap naturally. Early years (cooler) appear
blue; recent years (warmer) appear red.
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

tr.plot_line(
    figsize=(10, 10),
    title='Air temperature at ICOS Ecosystem Station Davos (2013-2022)',
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
