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
    xcol='Tair_f',  # Driver variable (air temperature)
    ycol='NEE_CUT_REF_f',  # Response variable (net ecosystem productivity)
    n_bins=100,  # Number of quantile-based bins along x
    bins_agg='median',  # Aggregate y within each bin using median
    rwinsize=0.1,  # Rolling window = 10% of total bins
    ragg='mean',  # Smooth bin medians with rolling mean
    define_optimum='min',  # Minimum NEE = maximum CO2 uptake
    threshold=0.95,  # Optimum range = bins within top 5% of curve amplitude
    prominence_threshold=1.0,  # Warn if peak is less than 1 std above curve mean
)

# Calculate optimum range
optrange.find_optimum()

# %%
# Results
# ^^^^^^^

results = optrange.results_optrange
print(f"Optimum temperature range: {results['optimum_xstart']:.2f} to {results['optimum_xend']:.2f} C")
print(f"Mean NEE in optimum range: {results['optimum_ymean']:.3f} umol/m2/s")
print(f"Optimum prominent: {results['is_optimum_prominent']}  (prominence={results['optimum_prominence']:.2f})")

# Show percentage of data in optimum range
vals_in_range = results['vals_in_optimum_range_df']
print("\nPercentage of data in optimum range (by year):")
print(vals_in_range[['vals_inoptimum_perc']])

# %%
# Visualization
# ^^^^^^^^^^^^^

optrange.showfig(
    xunit='degC',            # x-axis unit label: "Tair_f [degC]"
    yunit='umol m-2 s-1',   # y-axis unit label: "NEE_CUT_REF_f [umol m-2 s-1]"
    # xlabel='Air temperature',  # override full x label (optional)
    # ylabel='NEE',              # override full y label (optional)
)
