"""
=======================
Diel Cycle Analysis
=======================

Visualize diurnal (daily) cycles separated by month or season.
Shows hourly mean values and seasonal changes in daily patterns.

Best for: Understanding diurnal cycles and seasonal variation in fluxes
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
series = df['NEE_CUT_REF_f'].copy()

print(f"Loaded {len(series)} records from {df.index[0].date()} to {df.index[-1].date()}")

# %%
# Diel cycle by month
# ^^^^^^^^^^^^^^^^^^^
#
# Show diurnal cycles separated by month to reveal seasonal patterns. The plot
# chrome (title + y-axis units) is bundled into a shared
# ``dv.plotting.FormatStyle``; per-plot rendering choices (here ``each_month``
# and the legend column count) stay as direct ``plot()`` arguments.

dc = dv.plotting.DielCycle(series=series)

style = dv.plotting.FormatStyle(
    title=r'$\mathrm{Mean\ CO_2\ flux\ (2013-2024)}$',
    yunits=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
    legend_ncol=2,
)

dc.plot(ax=None, format_style=style, each_month=True)

print("\nPlotted diel cycle by month showing seasonal variation")

# %%
# Reusing one style across plots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same ``FormatStyle`` can be handed to several ``plot()`` calls so they
# share an identical look. Here the per-month style above is reused for an
# all-data diel cycle (a single mean curve), proving the chrome is defined once.

dc.plot(ax=None, format_style=style, each_month=False)

print("\nPlotted all-data diel cycle reusing the same FormatStyle")

# %%
# Statistics
# ^^^^^^^^^^

print(f"\nNEE flux statistics:")
print(f"  Mean: {series.mean():.2f}")
print(f"  Std dev: {series.std():.2f}")
print(f"  Min: {series.min():.2f}")
print(f"  Max: {series.max():.2f}")
