"""
============================
Advanced Heatmap Plots
============================

Year-month aggregation heatmaps and multi-variable side-by-side comparison.

Best for: Seasonal patterns, comparing multiple variables across dimensions
"""

import matplotlib.pyplot as plt
import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# HeatmapYearMonth - aggregation by month
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Yearly data aggregated by month with heatmap coloring.

series_temp = df['Tair_f'].copy()
series_temp = series_temp.dropna()

hm = dv.plot_heatmap_year_month(
    series=series_temp,
    ax_orientation="horizontal",       # Months on y-axis, years on x-axis
    ranks=False,                        # Use actual values (not ranks)
    show_values=True,                  # Display values on cells
    zlabel="°C",                        # Colorbar label
    vmin=None,                         # Auto min
    vmax=None,                         # Auto max
    cmap='RdYlBu_r'                    # Colormap
)
hm.show()

print("Plotted HeatmapYearMonth with aggregation")

# %%
# Multiple heatmaps side-by-side
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare three different variables in one figure using datetime heatmaps.

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
    vmin=-10,                          # Minimum color value
    vmax=10,                           # Maximum color value
    ax_orientation="horizontal",       # Time on y-axis
    show_values=False,                 # Don't show cell values
    cmap='RdBu_r'                      # Colormap
).plot()

dv.plot_heatmap_datetime(
    ax=axes[1],
    series=series_tair,
    zlabel="°C",
    vmin=-10,
    vmax=30,
    ax_orientation="horizontal",
    show_values=False,
    cmap='RdYlBu_r'
).plot()

dv.plot_heatmap_datetime(
    ax=axes[2],
    series=series_le,
    zlabel=r"$\mathrm{W\ m^{-2}}$",
    vmin=0,
    vmax=400,
    ax_orientation="horizontal",
    show_values=False,
    cmap='YlOrRd'
).plot()

axes[0].set_title("NEE (CO₂ flux)")
axes[1].set_title("Air temperature")
axes[2].set_title("Latent heat flux")

fig.show()

print("Plotted three variables side-by-side")
