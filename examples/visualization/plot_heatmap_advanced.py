"""
============================
Advanced Heatmap Plots
============================

Year-month aggregation heatmaps with multiple aggregation methods, and
multi-variable side-by-side comparison.

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

hm = dv.plotting.HeatmapYearMonth(
    series=series_temp,
    ax_orientation="vertical",  # Data computation: months on y-axis, years on x-axis
    ranks=False  # Data computation: use actual values (not ranks)
)
hm.plot(
    ax=None,  # Create new figure
    show_values=True,  # Display values on cells
    zlabel="°C",  # Colorbar label
    vmin=None,  # Auto min
    vmax=None,  # Auto max
    cmap='RdYlBu_r'  # Colormap
)

print("Plotted HeatmapYearMonth with aggregation")

# %%
# HeatmapYearMonth - comparing aggregation methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same data looks different depending on which statistic fills each cell.
# Mean shows the typical month; max reveals the extremes; std shows
# how variable each month is across years.

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, agg, cmap, label in [
    (axes[0], 'mean', 'RdYlBu_r', 'Mean °C'),
    (axes[1], 'max', 'YlOrRd', 'Max °C'),
    (axes[2], 'std', 'Blues', 'Std dev °C'),
]:
    dv.plotting.HeatmapYearMonth(
        series=series_temp,
        ax_orientation="vertical",
        agg=agg,  # aggregation applied to each year/month cell
        ranks=False
    ).plot(
        ax=ax,
        show_values=True,
        zlabel=label,  # colorbar label stays a direct kwarg (not FormatStyle chrome)
        cmap=cmap,
        # The title is shared plot chrome, so it goes through FormatStyle.
        format_style=dv.plotting.FormatStyle(title=f"agg='{agg}'")
    )

fig.show()
print("Plotted HeatmapYearMonth with mean / max / std aggregation")

# %%
# HeatmapYearMonth - ranks
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ranks=True replaces values with their rank order within each month column.
# Rank 1 = highest value for that month across all years.
# Useful for spotting record years without being distracted by absolute values.

hm_ranks = dv.plotting.HeatmapYearMonth(
    series=series_temp,
    ax_orientation="vertical",
    agg='mean',
    ranks=True  # rank within each month column
)
hm_ranks.plot(
    ax=None,
    show_values=True,
    zlabel="rank",
    cmap='RdYlGn_r'  # rank 1 (warmest) in red
)
print("Plotted HeatmapYearMonth with ranks")

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

# One FormatStyle, reused across all three panels: building it once keeps the
# tick/label font sizing identical, so the only per-panel difference is the
# title (folded on via .merged()) and the colorbar (a direct kwarg).
panel_style = dv.plotting.FormatStyle(axlabel_fontsize=12, ticks_fontsize=10)

dv.plotting.HeatmapDateTime(
    series=series_nee,
    ax_orientation="vertical"  # Data: time on y-axis
).plot(
    ax=axes[0],  # Render on first subplot
    zlabel=r"$\mathrm{\mu mol\ CO_2\ m^{-2}\ s^{-1}}$",  # colorbar label stays direct
    vmin=-10,  # Minimum color value
    vmax=10,  # Maximum color value
    show_values=False,  # Don't show cell values
    cmap='RdBu_r',  # Colormap
    format_style=panel_style.merged(title="NEE (CO₂ flux)")
)

dv.plotting.HeatmapDateTime(
    series=series_tair,
    ax_orientation="vertical"
).plot(
    ax=axes[1],  # Render on second subplot
    zlabel="°C",
    vmin=-10,
    vmax=30,
    show_values=False,
    cmap='RdYlBu_r',
    format_style=panel_style.merged(title="Air temperature")
)

dv.plotting.HeatmapDateTime(
    series=series_le,
    ax_orientation="vertical"
).plot(
    ax=axes[2],  # Render on third subplot
    zlabel=r"$\mathrm{W\ m^{-2}}$",
    vmin=0,
    vmax=400,
    show_values=False,
    cmap='YlOrRd',
    format_style=panel_style.merged(title="Latent heat flux")
)

fig.show()

print("Plotted three variables side-by-side")
