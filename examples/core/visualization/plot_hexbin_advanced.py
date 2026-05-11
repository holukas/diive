"""
=======================
Advanced Hexbin Plots
=======================

2D hexagonal binning with absolute values and value overlay visualization.

Best for: Detailed density visualization with actual values and custom aggregation
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
# Hexbin with absolute values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use original variable values on axes with mean aggregation in each bin.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=False,              # Use original data values (not percentile)
    gridsize=15,                       # Number of hexagons per side
    xlabel='Air temperature (°C)',
    ylabel='Vapor pressure deficit (hPa)',
    zlabel='Mean NEE',
    cmap='RdYlBu_r',                   # Colormap name
    reduce_C_function=np.mean,         # Use mean aggregation (not median)
    show_values=False,                 # Don't show values on hexagons
    cb_digits_after_comma=0            # Colorbar decimal places
)
hm.show()

print("Plotted hexbin with absolute values and mean aggregation")

# %%
# Hexbin with value overlay
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display aggregated values directly on hexagon centers for detailed readability.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,               # Use percentile scale (0-100)
    gridsize=20,                       # Number of hexagons per side
    xlabel='Temperature (percentile)',
    ylabel='VPD (percentile)',
    zlabel='Mean NEE',
    cmap='RdYlBu_r',                   # Colormap name
    reduce_C_function=np.mean,         # Use mean aggregation
    show_values=True,                  # Display values on hexagon centers
    show_values_fontsize=8,            # Font size for overlay values
    show_values_n_dec_places=0,        # Decimal places in overlay
    cb_digits_after_comma=1            # Colorbar decimal places
)
hm.show()

print("Plotted hexbin with value overlays on hexagon centers")
