"""
============================
Basic 2D Scatter Plot
============================

Simple scatter plot of two variables showing their relationship.

Best for: Exploring variable relationships, quick correlation checks
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
df_2022 = df.loc[df.index.year == 2022].copy()

print(f"Loaded {len(df_2022)} records for 2022")

# %%
# Create scatter plot
# ^^^^^^^^^^^^^^^^^^^
#
# Chrome (title, axis labels, units, font sizes) lives in a shared
# ``FormatStyle``; the colorbar settings (``cmap``, ``show_colorbar``) and the
# data ranges (``xlim``, ``ylim``) stay as direct ``plot()`` arguments because
# they are not part of the cross-cutting chrome.

scatter = dv.plotting.ScatterXY(
    x=df_2022['Tair_f'],
    y=df_2022['VPD_f'],
    z=None,  # No third variable
    nbins=0,  # No bin aggregation
    binagg='median'  # (ignored if nbins=0)
)
scatter.plot(
    ax=None,  # Create new figure
    format_style=dv.plotting.FormatStyle(
        title='Temperature vs VPD relationship',
        xlabel='Air temperature (°C)',
        ylabel='VPD (hPa)',
    ),
    xlim=None,  # Auto data range
    ylim=None,  # Auto data range
    cmap='viridis',  # Default colormap (used when z is present)
    show_colorbar=True,  # Show colorbar if z present
)

print("Plotted temperature vs VPD scatter")

# %%
# Calculate correlation
# ^^^^^^^^^^^^^^^^^^^^^

correlation = df_2022['Tair_f'].corr(df_2022['VPD_f'])
print(f"\nCorrelation (Tair-VPD): {correlation:.3f}")
