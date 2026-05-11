"""
============================
Basic Hexbin Plot (Percentile)
============================

2D hexagonal binning with percentile normalization for standardized comparison.

Best for: Comparing relationships with standardized axes (0-100 percentile range)
"""

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
    normalize_axes=True,               # Use percentile scale (0-100)
    gridsize=11,                       # Number of hexagons per side
    xlabel='Air temperature (percentile)',
    ylabel='Vapor pressure deficit (percentile)',
    zlabel='NEE aggregated',
    cmap='RdYlBu_r',                   # Colormap name
    show_values=False,                 # Don't show values on hexagons
    cb_digits_after_comma=1            # Colorbar decimal places
)
hm.show()

print("Plotted hexbin with percentile normalization")
