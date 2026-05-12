"""
=================================
Optimum Range Detection
=================================

Find optimal driver variable ranges for optimized response.

Demonstrates finding the air temperature range where net ecosystem
productivity (NEE) is optimized during daytime, with analysis of data
distribution within, above, and below the optimum range.

Best for: Identifying operating conditions for ecosystem processes
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

df_orig = dv.load_exampledata_parquet()

# Filter to daytime data (solar radiation > 20 W/m^2)
df = df_orig.copy()
df = df.loc[df['Rg_f'] > 20]

# %%
# Find optimum temperature range
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Find optimum temperature range for maximum CO2 uptake (minimum NEE)
optrange = dv.find_optimum_range(
    df=df,
    xcol='Tair_f',         # Driver variable (air temperature)
    ycol='NEE_CUT_REF_f',  # Response variable (net ecosystem productivity)
    n_bins=100,            # Number of bins for x-axis
    define_optimum='min'   # Minimum NEE = maximum CO2 uptake
)

# Calculate optimum range
optrange.find_optimum()

# %%
# Results
# ^^^^^^^

results = optrange.results_optrange
print(f"Optimum temperature range: {results['optimum_xstart']:.2f} to {results['optimum_xend']:.2f} C")
print(f"Mean NEE in optimum range: {results['optimum_ymean']:.3f} umol/m2/s")

# Show percentage of data in optimum range
vals_in_range = results['vals_in_optimum_range_df']
print("\nPercentage of data in optimum range (by year):")
print(vals_in_range[['vals_inoptimum_perc']])

# %%
# Visualization
# ^^^^^^^^^^^^^

optrange.showfig()
