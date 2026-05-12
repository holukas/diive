"""
=============================
Advanced Ridge Line Analysis
=============================

Multiple ridge line examples with different time groupings and styling options.
Compare distribution patterns across months, years, and seasons.

Best for: Detailed temporal evolution, comparing multiple variables, publication-quality plots
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# Monthly ridge line plot
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Show air temperature distribution for each month.
# Monthly grouping provides more granular temporal resolution than weekly,
# ideal for detecting seasonal shifts in temperature distributions.

df_2022 = df.loc[df.index.year == 2022].copy()
series_monthly = df_2022['Tair_f'].copy()

rp = dv.plot_ridgeline(
    series=series_monthly
)
rp.plot(
    how='monthly',
    xlabel=r'Air temperature (°C)',
    fig_width=10,
    fig_height=8,
    shade_percentile=0.5,
    show_mean_line=False
)

print("\nPlotted monthly ridge lines for 2022")
print("Monthly resolution clearly shows temperature distribution evolution from winter to summer")
