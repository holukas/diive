"""
=======================
Advanced Hexbin Plots
=======================

2D hexagonal binning with absolute values and value overlay visualization.

Best for: Detailed density visualization with actual values and custom aggregation
"""

import matplotlib.pyplot as plt
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

hm = dv.plotting.HexbinPlot(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=False,              # Data: use original data values (not percentile)
    gridsize=15,                       # Data: number of hexagons per side
    xlabel='Air temperature (°C)',
    ylabel='Vapor pressure deficit (hPa)',
    zlabel='Mean NEE',
    reduce_C_function=np.mean          # Data: use mean aggregation (not median)
)
hm.plot(
    ax=None,                           # Create new figure
    cmap='RdYlBu_r',                   # Styling: colormap name
    show_values=False,                 # Styling: don't show values on hexagons
    cb_digits_after_comma=0            # Styling: colorbar decimal places
)

plt.show(block=False)

print("Plotted hexbin with absolute values and mean aggregation")

# %%
# Hexbin with value overlay
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display aggregated values directly on hexagon centers for detailed readability.

hm = dv.plotting.HexbinPlot(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,               # Data: use percentile scale (0-100)
    gridsize=20,                       # Data: number of hexagons per side
    xlabel='Temperature (percentile)',
    ylabel='VPD (percentile)',
    zlabel='Mean NEE',
    reduce_C_function=np.mean          # Data: use mean aggregation
)
hm.plot(
    ax=None,                           # Create new figure
    cmap='RdYlBu_r',                   # Styling: colormap name
    show_values=True,                  # Styling: display values on hexagon centers
    show_values_fontsize=8,            # Styling: font size for overlay values
    show_values_n_dec_places=0,        # Styling: decimal places in overlay
    cb_digits_after_comma=1            # Styling: colorbar decimal places
)

plt.show(block=False)

print("Plotted hexbin with value overlays on hexagon centers")
