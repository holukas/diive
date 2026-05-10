"""
========================================
Linear Interpolation - Conservative Limit
========================================

Gap-fill time series using linear interpolation with conservative gap size limit.

Demonstrates linear interpolation with limit=1, filling only the smallest gaps
(isolated single missing values). Useful when you want to preserve larger data
gaps for other gap-filling methods (e.g., Machine Learning).

Best for: Pre-processing step to remove only isolated missing values.
"""

# %%
# Linear interpolation with conservative limit
# =============================================
#
# Demonstrates linear interpolation with limit=1, filling only the smallest gaps
# (isolated single missing values). Useful when you want to preserve larger data
# gaps for other gap-filling methods (e.g., Machine Learning).

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2022].copy()

series = df['NEE_CUT_REF_orig'].copy()

print(f"Data loaded: {len(series)} records from {series.index.min().date()} to {series.index.max().date()}")
print(f"Missing values before: {series.isnull().sum()}")

# %%
# Conservative gap-filling (limit=1)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Fill only isolated missing values (gaps of 1 record).

series_gapfilled = dv.linear_interpolation(series=series, limit=1, verbose=True)

print(f"Missing values after: {series_gapfilled.isnull().sum()}")

# %%
# Visualization: before and after
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = plt.figure(facecolor='white', figsize=(16, 9))
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
ax_input = fig.add_subplot(gs[0, 0])
ax_output = fig.add_subplot(gs[0, 1])

dv.plot_heatmap_datetime(ax=ax_input, series=series).plot()
dv.plot_heatmap_datetime(ax=ax_output, series=series_gapfilled).plot()

ax_input.set_title("Observed Data (with gaps)", color='black', fontsize=12, fontweight='bold')
ax_output.set_title("Gap-Filled (limit=1, conservative)", color='black', fontsize=12, fontweight='bold')
ax_input.tick_params(left=True, right=False, top=False, bottom=True,
                     labelleft=False, labelright=False, labeltop=False, labelbottom=True)
ax_output.tick_params(left=True, right=False, top=False, bottom=True,
                      labelleft=False, labelright=False, labeltop=False, labelbottom=True)

fig.show()

print("✓ Linear interpolation (conservative) complete.")
