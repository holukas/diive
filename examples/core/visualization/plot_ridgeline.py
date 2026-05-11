"""
=======================
Ridge Line Plotting
=======================

Kernel density estimate plots stacked by time period (weekly, monthly, yearly)
to show distribution changes over time.

Best for: Visualizing seasonal variation, temporal evolution of distributions
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
print(f"Loaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# Weekly ridge line plot
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Show air temperature distribution for each week of 2019.

df_2019 = df.loc[(df.index.year >= 2019) & (df.index.year <= 2019)].copy()
series_weekly = df_2019['Tair_f'].copy()

rp = dv.plot_ridgeline(series=series_weekly)
rp.plot(
    how='weekly',
    xlabel=r'Air temperature (°C)',
    fig_width=8,
    fig_height=10,
    shade_percentile=0.5,
    show_mean_line=False
)

print("\nPlotted weekly ridge lines for 2019")

# %%
# Monthly ridge line plot
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Show air temperature distribution for each month of 2022.

df_2022 = df.loc[df.index.year == 2022].copy()
series_monthly = df_2022['Tair_f'].copy()

rp = dv.plot_ridgeline(series=series_monthly)
rp.plot(
    how='monthly',
    xlabel=r'Air temperature (°C)',
    fig_width=10,
    fig_height=8,
    shade_percentile=0.5
)

print("Plotted monthly ridge lines for 2022")
