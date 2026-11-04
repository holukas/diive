"""
=============================
Advanced Ridge Line Analysis
=============================

Ridge line plot with monthly grouping and a figure title.
Compare distribution patterns across the months of one year.

Best for: Seasonal shifts in a distribution, publication-quality plots
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
# Monthly grouping gives fewer and broader ridges than weekly grouping. This
# makes seasonal shifts in the temperature distribution easy to see.

df_2022 = df.loc[df.index.year == 2022].copy()
series_monthly = df_2022['Tair_f'].copy()

# The chrome that RidgeLinePlot honours -- the bottom x-axis label and the
# figure title -- lives in a shared FormatStyle, so the same look could be
# reused across several ridgeline figures. Per-ridge layout (fig size, shading,
# mean line) stays a direct plot() argument.
style = dv.plotting.FormatStyle(
    xlabel=r'Air temperature (°C)',
    title='Monthly air temperature distribution (2022)'
)

rp = dv.plotting.RidgeLinePlot(
    series=series_monthly
)
rp.plot(
    how='monthly',
    fig_width=10,
    fig_height=8,
    shade_percentile=0.5,
    show_mean_line=False,
    format_style=style
)

print("\nPlotted monthly ridge lines for 2022")
print("Monthly resolution clearly shows temperature distribution evolution from winter to summer")
