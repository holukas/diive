"""
==============================
Heatmap Visualization
==============================

Temporal heatmaps for time series visualization: date × time-of-day grids
and year × month aggregation heatmaps with customizable orientations.

Best for: Visualizing temporal patterns, identifying systematic trends, comparing variables
"""

import matplotlib.pyplot as plt
import numpy as np
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
    vmin=-10,
    vmax=10,
    ax_orientation="vertical"
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
    vmin=-10,
    vmax=10,
    ax_orientation="horizontal"
)
hm.show()

print("Plotted HeatmapDateTime in horizontal orientation")

# %%
# HeatmapYearMonth - aggregation by month
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Yearly data aggregated by month with heatmap coloring.

series_temp = df['Tair_f'].copy()
series_temp = series_temp.dropna()

hm = dv.plot_heatmap_year_month(
    series=series_temp,
    ax_orientation="horizontal",
    ranks=False,
    show_values=True,
    zlabel="°C"
)
hm.show()

print("Plotted HeatmapYearMonth with aggregation")

# %%
# Multiple heatmaps side-by-side
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare three different variables in one figure.

series_nee = df['NEE_CUT_REF_f'].copy()
series_tair = df['Tair_f'].copy()
series_le = df['LE_f'].copy()

# Filter to recent years for cleaner visualization
locs = series_nee.index.year >= 2020
series_nee = series_nee.loc[locs]
series_tair = series_tair.loc[locs]
series_le = series_le.loc[locs]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

dv.plot_heatmap_datetime(
    ax=axes[0],
    series=series_nee,
    zlabel=r"$\mathrm{\mu mol\ CO_2\ m^{-2}\ s^{-1}}$",
    vmin=-10,
    vmax=10
).plot()

dv.plot_heatmap_datetime(
    ax=axes[1],
    series=series_tair,
    zlabel="°C",
    vmin=-10,
    vmax=30
).plot()

dv.plot_heatmap_datetime(
    ax=axes[2],
    series=series_le,
    zlabel=r"$\mathrm{W\ m^{-2}}$",
    vmin=0,
    vmax=400
).plot()

axes[0].set_title("NEE (CO₂ flux)")
axes[1].set_title("Air temperature")
axes[2].set_title("Latent heat flux")

fig.show()

print("Plotted three variables side-by-side")
