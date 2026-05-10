"""
============================================
Photosynthetic Decoupling Analysis
============================================

Stratified analysis of photosynthetic response across temperature gradients.

Demonstrates stratified binning analysis: how does net ecosystem productivity
(NEE) respond to vapor pressure deficit (VPD) across different temperature
classes? Reveals environmental controls on ecosystem carbon uptake.

Best for: Understanding multi-variable interactions in photosynthesis
"""

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

import diive as dv

data_df = dv.load_exampledata_parquet()

# Use summer months only
data_df = data_df.loc[(data_df.index.month >= 6) & (data_df.index.month <= 9)].copy()

# Select variables
vpd_col = 'VPD_f'
nee_col = 'NEE_CUT_REF_f'
ta_col = 'Tair_f'
gpp_col = 'GPP_DT_CUT_REF'
swin_col = 'Rg_f'

df = data_df[[nee_col, gpp_col, ta_col, vpd_col, swin_col]].copy()

# Filter to daytime only (solar radiation > 50 W/m^2)
daytime_locs = (df[swin_col] > 50) & (df[ta_col] > 5)
df = df[daytime_locs].copy()

# Rename for clarity
rename_dict = {
    ta_col: 'air_temperature',
    vpd_col: 'vapor_pressure_deficit',
    nee_col: 'net_ecosystem_productivity'
}
df = df.rename(columns=rename_dict, inplace=False)

# Prepare data - convert to CO2 uptake (positive = uptake)
ta_col = 'air_temperature'
vpd_col = 'vapor_pressure_deficit'
nee_col = 'net_ecosystem_productivity'
df = df[[ta_col, vpd_col, nee_col]].copy()
df[nee_col] = df[nee_col].multiply(-1)

# %%
# Stratified analysis
# ^^^^^^^^^^^^^^^^^^^

# Analyze with stratified binning across temperature classes
analysis = dv.stratified_analysis(
    df=df,
    zvar=ta_col,  # Stratification variable (temperature)
    xvar=vpd_col,  # X-axis (vapor pressure deficit)
    yvar=nee_col,  # Y-axis (CO2 uptake)
    n_bins_z=10,  # Temperature strata
    n_bins_x=5,   # VPD bins
    conversion=False
)

# Calculate bins and display results
analysis.calcbins()

# %%
# Results and interpretation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Access results as dictionary
binaggs = analysis.get_binaggs()
first = next(iter(binaggs))
print("First temperature bin sample:")
print(binaggs[first])

# Get all results as a single dataframe
print("\nAll stratified analysis results:")
print(analysis.results)

# %%
# Visualization
# ^^^^^^^^^^^^^

# Show decoupling patterns across temperature strata
analysis.showplot_decoupling_sbm(marker='o', emphasize_lines=True)
