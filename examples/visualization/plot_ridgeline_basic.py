"""
=======================
Ridge Line Plotting
=======================

Kernel density estimate plots stacked by time period
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
# Show air temperature distribution for each week.
# Ridge lines display kernel density estimates stacked vertically,
# making it easy to compare distribution shapes across time periods.

df_2019 = df.loc[(df.index.year >= 2019) & (df.index.year <= 2019)].copy()
series_weekly = df_2019['Tair_f'].copy()

# The chrome (here just the x-axis label) goes into a shared FormatStyle;
# the ridgeline-specific rendering toggles stay as direct plot() arguments.
style = dv.plotting.FormatStyle(
    xlabel='Air temperature',
    xunits='(°C)',
)

rp = dv.plotting.RidgeLinePlot(
    series=series_weekly
)
rp.plot(
    how='weekly',
    fig_width=8,
    fig_height=10,
    shade_percentile=0.5,
    show_mean_line=False,
    format_style=style
)

print("\nPlotted weekly ridge lines for 2019")
print(f"Weekly distribution shapes reveal seasonal temperature patterns")
