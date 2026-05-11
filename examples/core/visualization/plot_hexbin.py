"""
===================
Hexbin Plotting
===================

2D hexagonal binning aggregation plots for relationship analysis
between variables. Aggregates values in hexagonal bins for density visualization.

Best for: Visualizing 2D relationships and finding patterns in dense data
"""

import numpy as np
import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

# Select growing season for cleaner visualization
df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()
data = df[['Tair_f', 'VPD_f', 'NEE_CUT_REF_f']].dropna()

print(f"Loaded {len(data)} growing season records")

# %%
# Hexbin with percentile normalization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Normalize axes to 0-100 percentile scale for standardized comparison.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,
    gridsize=11,
    xlabel='Air temperature (percentile)',
    ylabel='Vapor pressure deficit (percentile)',
    zlabel='NEE aggregated',
    cmap='RdYlBu_r'
)
hm.show()

print("Plotted hexbin with percentile normalization")

# %%
# Hexbin with absolute values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use original variable values on axes with mean aggregation in each bin.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=False,
    gridsize=15,
    reduce_C_function=np.mean,
    xlabel='Air temperature (°C)',
    ylabel='Vapor pressure deficit (hPa)',
    zlabel='Mean NEE',
    cb_digits_after_comma=0,
    cmap='RdYlBu_r'
)
hm.show()

print("Plotted hexbin with absolute values")

# %%
# Hexbin with value overlay
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display aggregated values directly on hexagon centers for readability.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,
    gridsize=20,
    reduce_C_function=np.mean,
    xlabel='Temperature (percentile)',
    ylabel='VPD (percentile)',
    zlabel='Mean NEE',
    show_values=True,
    show_values_fontsize=8,
    show_values_n_dec_places=0,
    cmap='RdYlBu_r'
)
hm.show()

print("Plotted hexbin with value overlays")
