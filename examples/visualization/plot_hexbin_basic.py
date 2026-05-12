"""
============================
Basic Hexbin Plot (Percentile)
============================

2D hexagonal binning with percentile normalization for standardized comparison.

Best for: Comparing relationships with standardized axes (0-100 percentile range)
"""

import matplotlib.pyplot as plt
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
# Create hexbin with percentile normalization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Normalize axes to 0-100 percentile scale for standardized comparison
# across different variables.

hm = dv.plot_hexbin(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,               # Data: use percentile scale (0-100)
    gridsize=11,                       # Data: number of hexagons per side
    xlabel='Air temperature (percentile)',
    ylabel='Vapor pressure deficit (percentile)',
    zlabel='NEE aggregated'
)
hm.plot(
    ax=None,                           # Create new figure
    cmap='RdYlBu_r',                   # Styling: colormap name
    show_values=False,                 # Styling: don't show values on hexagons
    cb_digits_after_comma=1            # Styling: colorbar decimal places
)

plt.show(block=False)

print("Plotted hexbin with percentile normalization")
