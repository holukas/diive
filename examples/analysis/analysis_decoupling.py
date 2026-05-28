"""
=================
Decoupling Analysis with Sorting Bins
=================

Stratified binning to reveal how radiation response changes across temperature ranges.

How does ecosystem radiation response depend on temperature? This example bins data
by vapor pressure deficit (VPD) within separate temperature classes, showing whether
the relationship stays stable or changes as temperature increases.

Best for: Understanding how environmental controls interact
"""

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

import diive as dv

# Load the dataset
data_df = dv.load_exampledata_parquet()

# Keep summer months (June-September is typical for vegetation growth)
data_df = data_df.loc[(data_df.index.month >= 6) & (data_df.index.month <= 9)].copy()

# Select variables
vpd_col = 'VPD_f'
tair_col = 'Tair_f'
radiation_col = 'Rg_f'

df = data_df[[vpd_col, tair_col, radiation_col]].copy()

# Keep daytime only: radiation > 20 W/m^2 and temperature > 0 C
# (These thresholds define when the ecosystem is actually functioning)
daytime_locs = (df[radiation_col] > 20) & (df[tair_col] > 0)
df = df[daytime_locs].copy()

print(f"Loaded {len(df)} daytime records from summer")

# %%
# Bin data by VPD within temperature strata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# The idea: group data into temperature bins, then within each bin,
# bin by VPD. This shows if the radiation-VPD relationship changes
# as temperature increases (decoupling).

analysis = dv.analysis.StratifiedAnalysis(
    df=df,
    xvar=vpd_col,           # Bin by VPD on x-axis
    yvar=radiation_col,     # Show radiation on y-axis
    zvar=tair_col,          # Stratify by temperature
    n_bins_z=5,             # 5 temperature bins
    n_bins_x=10,            # 10 VPD bins within each temperature class
    conversion=None,        # No z-score conversion
    agg='median'            # Use median for aggregation
)

analysis.calcbins()

print("Binning complete")

# %%
# Access and examine results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Results are stored as a dictionary with temperature medians as keys
bin_results = analysis.get_binaggs()

# List all temperature classes (as strings of temperature medians)
temp_classes = list(bin_results.keys())
print(f"\nTemperature bins (medians in C): {temp_classes}")

# Look at the warmest bin
warmest = bin_results[temp_classes[-1]]
print(f"\nResults for warmest class ({temp_classes[-1]} C):")
print(warmest)

# %%
# Visualization
# ^^^^^^^^^^^^^

# Plot the stratified relationship
analysis.showplot_decoupling_sbm(marker='o', emphasize_lines=True)
