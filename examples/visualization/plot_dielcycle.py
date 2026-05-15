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
# Show diurnal cycles separated by month to reveal seasonal patterns.

dc = dv.plot_diel_cycle(series=series)
title = r'$\mathrm{Mean\ CO_2\ flux\ (2013-2024)}$'
units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'

dc.plot(ax=None, title=title, txt_ylabel_units=units, each_month=True, legend_n_col=2)

print("\nPlotted diel cycle by month showing seasonal variation")

# %%
# Statistics
# ^^^^^^^^^^

print(f"\nNEE flux statistics:")
print(f"  Mean: {series.mean():.2f}")
print(f"  Std dev: {series.std():.2f}")
print(f"  Min: {series.min():.2f}")
print(f"  Max: {series.max():.2f}")
