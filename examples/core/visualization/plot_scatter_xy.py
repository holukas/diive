"""
====================
Scatter Plotting
====================

2D scatter plots with optional third variable as color coding.
Includes bin aggregation and regression overlays.

Best for: Exploring variable relationships, identifying correlations
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
df_2022 = df.loc[df.index.year == 2022].copy()

print(f"Loaded {len(df_2022)} records for 2022")

# %%
# Basic scatter: Temperature vs VPD
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

scatter = dv.plot_scatter_xy(
    x=df_2022['Tair_f'],
    y=df_2022['VPD_f']
)
scatter.plot(
    xlabel='Air temperature (°C)',
    ylabel='VPD (hPa)',
    title='Temperature vs VPD relationship'
)

print("Plotted temperature vs VPD scatter")

# %%
# 3D scatter: NEE vs VPD colored by radiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Third variable (radiation) shown as point color.

scatter = dv.plot_scatter_xy(
    x=df_2022['VPD_f'],
    y=df_2022['NEE_CUT_REF_f'],
    z=df_2022['Rg_f'],
    title='CO2 flux response to VPD'
)
scatter.plot(
    xlabel='VPD (hPa)',
    ylabel=r'NEE ($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    zlabel=r'Radiation ($\mathrm{W\ m^{-2}}$)',
    cmap='plasma',
    show_colorbar=True
)

print("Plotted NEE vs VPD colored by radiation")

# %%
# Statistics
# ^^^^^^^^^^

print(f"\nVariable correlations:")
print(f"  Tair-VPD: {df_2022['Tair_f'].corr(df_2022['VPD_f']):.3f}")
print(f"  NEE-VPD: {df_2022['NEE_CUT_REF_f'].corr(df_2022['VPD_f']):.3f}")
