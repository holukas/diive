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

# Chrome lives in a shared FormatStyle (title, axis labels, and the colorbar
# label ``zlabel``); ``cmap``/``show_colorbar`` are rendering-specific and stay
# on plot(). The two scatters below colour by different variables, so each
# overrides just ``zlabel`` via ``style.merged(...)`` while keeping the look.
style = dv.plotting.FormatStyle(
    title='CO2 flux response to VPD',
    xlabel='VPD (hPa)',
    ylabel=r'NEE ($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
)

scatter = dv.plotting.ScatterXY(
    x=df_2022['VPD_f'],
    y=df_2022['NEE_CUT_REF_f'],
    z=df_2022['Rg_f'],  # Color by radiation
    nbins=10,  # Aggregate into 10 bins
    binagg='median'  # Use median for bin aggregation
)
scatter.plot(
    ax=None,  # Create new figure
    format_style=style.merged(zlabel=r'Radiation ($\mathrm{W\ m^{-2}}$)'),  # colorbar label
    xlim=None,  # Auto data range
    ylim=None,  # Auto data range
    cmap='plasma',  # Colormap for z variable
    show_colorbar=True,  # Display colorbar
)

print("Plotted NEE vs VPD colored by radiation with bin aggregation")

# %%
# Reuse the same style on a second scatter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The key benefit of a shared FormatStyle: build the chrome once, reuse it
# across plots. Here we colour NEE vs VPD by air temperature instead of
# radiation, but keep the identical title/label look via the same ``style``.

scatter_temp = dv.plotting.ScatterXY(
    x=df_2022['VPD_f'],
    y=df_2022['NEE_CUT_REF_f'],
    z=df_2022['Tair_f'],  # Color by air temperature
    nbins=10,
    binagg='median'
)
scatter_temp.plot(
    ax=None,
    format_style=style.merged(zlabel='Air temperature (°C)'),  # same chrome, new colorbar label
    cmap='plasma',
    show_colorbar=True,
)

print("Plotted the same relationship coloured by temperature, reusing the style")

# %%
# Calculate correlation
# ^^^^^^^^^^^^^^^^^^^^^

correlation = df_2022['NEE_CUT_REF_f'].corr(df_2022['VPD_f'])
print(f"\nCorrelation (NEE-VPD): {correlation:.3f}")
