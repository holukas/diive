"""
======================
Lagged Variable Variants
======================

Demonstrates creating lagged and lead variants of time series variables.
Lagged variants are used for correlation analysis, feature engineering,
and investigating temporal relationships between variables.

Best for: Discovering lag-lead relationships and creating features for ML models.
"""

# %%
# Basic Lagged Variants
# ^^^^^^^^^^^^^^^^^^^^^
#
# Create lagged variants (past and future values) for a subset of data.
# This shows how to use different lag ranges and step sizes.

import matplotlib.pyplot as plt

import diive as dv

# Load example data: select a subset for clear visualization
df = dv.load_exampledata_parquet()
locs = (df.index.year == 2022) & (df.index.month == 7) & (df.index.hour >= 10) & (df.index.hour <= 15)
df = df[locs].copy()
df = df[['Tair_f', 'Rg_f', 'NEE_CUT_REF_f']].copy()

print("Basic Lagged Variants")
print("=" * 50)

# Create lagged variants: [-2, -1, +1] (skip 0)
results = dv.features.lagged_variants(
    df=df,
    lag=[-2, 1],  # Range from -2 to +1
    stepsize=1,  # Step of 1 record
    exclude_cols=['NEE_CUT_REF_f'],  # Don't lag this variable
    verbose=True
)

print(f"\nOriginal columns: {list(df.columns)}")
print(f"Result columns: {list(results.columns)}")
print(f"\nFirst 5 rows of results:")
print(results.head())

# %%
# Larger Lag Ranges
# ^^^^^^^^^^^^^^^^^
#
# Create lagged variants with larger lag values and custom step sizes
# to investigate longer-term temporal relationships between variables.

# Load full month of data
df = dv.load_exampledata_parquet()
locs = (df.index.year == 2022) & (df.index.month == 7)
df = df[locs].copy()
df = df[['Tair_f', 'Rg_f']].copy()

print("\nLarger Lag Ranges")
print("=" * 50)

# Create lagged variants with larger lags and step size
# This creates lags: -24, -18, -12, -6 (skips 0) and +6, +12, +18
results = dv.features.lagged_variants(
    df=df,
    lag=[-24, 18],  # Range from -24 to +18 records (half-hourly = 12h to 9h)
    stepsize=6,  # Step of 6 records (3 hours for half-hourly data)
    exclude_cols=None,
    verbose=True
)

print(f"\nCreated {len(results.columns) - len(df.columns)} lagged columns")
print(f"Total columns now: {len(results.columns)}")

# %%
# Visualization of Lagged Variants
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize how lagged variants appear in a time series plot,
# highlighting the temporal offset created by lagging.

# Load and filter example data
df = dv.load_exampledata_parquet()
locs = (df.index.year == 2022) & (df.index.month == 7) & (df.index.hour >= 10) & (df.index.hour <= 15)
df = df[locs].copy()
df = df[['Tair_f', 'Rg_f']].copy()

# Create lagged variants
results = dv.features.lagged_variants(
    df=df,
    lag=[-2, 1],
    stepsize=1,
    exclude_cols=None,
    verbose=False
)

# Plot original and lagged variants
plotdf = results.head(10).copy()

fig = plt.figure(facecolor='white', figsize=(16, 6), constrained_layout=True)
ax1 = fig.add_subplot(111)

dv.plotting.TimeSeries(series=plotdf['Tair_f']).plot(ax=ax1, color='#1565C0')
dv.plotting.TimeSeries(series=plotdf['.Tair_f-1']).plot(ax=ax1, color='#D32F2F')
dv.plotting.TimeSeries(series=plotdf['.Tair_f-2']).plot(ax=ax1, color='#00BCD4')
dv.plotting.TimeSeries(series=plotdf['.Tair_f+1']).plot(ax=ax1, color='#AB47BC')

# Add manual legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1565C0', lw=2, label='Original (Tair_f)'),
    Line2D([0], [0], color='#D32F2F', lw=2, label='Lag -1'),
    Line2D([0], [0], color='#00BCD4', lw=2, label='Lag -2'),
    Line2D([0], [0], color='#AB47BC', lw=2, label='Lag +1'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax1.set_title("Lagged variants of air temperature", fontsize=14, fontweight='bold')
ax1.text(0.98, -0.25,
         "Note: Lagging creates gaps at start/end of dataset. Gaps are filled with nearest available value.",
         size=10, color='k', transform=ax1.transAxes,
         alpha=0.7, horizontalalignment='right', verticalalignment='top')

fig.show()
