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

scatter = dv.plotting.ScatterXY(
    x=df_2022['Tair_f'],
    y=df_2022['VPD_f'],
    z=None,  # No third variable
    nbins=0,  # No bin aggregation
    binagg='median'  # (ignored if nbins=0)
)
scatter.plot(
    ax=None,  # Create new figure
    xlabel='Air temperature (°C)',
    ylabel='VPD (hPa)',
    zlabel=None,  # No color bar label
    xunits=None,  # No units appended
    yunits=None,  # No units appended
    xlim=None,  # Auto data range
    ylim=None,  # Auto data range
    cmap='viridis',  # Default colormap
    show_colorbar=True,  # Show colorbar if z present
    title='Temperature vs VPD relationship'
)

print("Plotted temperature vs VPD scatter")

# %%
# Calculate correlation
# ^^^^^^^^^^^^^^^^^^^^^

correlation = df_2022['Tair_f'].corr(df_2022['VPD_f'])
print(f"\nCorrelation (Tair-VPD): {correlation:.3f}")
