"""
========================================
3D Scatter Plot with Color Coding
========================================

Scatter plot of two variables with third variable shown as point colors.
Includes bin aggregation with trend overlay.

Best for: Exploring three-variable relationships, identifying patterns colored by condition
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
df_2022 = df.loc[df.index.year == 2022].copy()

print(f"Loaded {len(df_2022)} records for 2022")

# %%
# Create colored scatter plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Third variable (radiation) controls the color of each point.
# Bin aggregation groups data by x-axis and overlays trend lines.

scatter = dv.plot_scatter_xy(
    x=df_2022['VPD_f'],
    y=df_2022['NEE_CUT_REF_f'],
    z=df_2022['Rg_f'],  # Color by radiation
    nbins=10,  # Aggregate into 10 bins
    binagg='median'  # Use median for bin aggregation
)
scatter.plot(
    ax=None,  # Create new figure
    xlabel='VPD (hPa)',
    ylabel=r'NEE ($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    zlabel=r'Radiation ($\mathrm{W\ m^{-2}}$)',
    xunits=None,  # No units appended to xlabel
    yunits=None,  # No units appended to ylabel
    xlim=None,  # Auto data range
    ylim=None,  # Auto data range
    cmap='plasma',  # Colormap for z variable
    show_colorbar=True,  # Display colorbar
    title='CO2 flux response to VPD'
)

print("Plotted NEE vs VPD colored by radiation with bin aggregation")

# %%
# Calculate correlation
# ^^^^^^^^^^^^^^^^^^^^^

correlation = df_2022['NEE_CUT_REF_f'].corr(df_2022['VPD_f'])
print(f"\nCorrelation (NEE-VPD): {correlation:.3f}")
