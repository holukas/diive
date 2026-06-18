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

hm = dv.plotting.HexbinPlot(
    x=data['Tair_f'],
    y=data['VPD_f'],
    z=data['NEE_CUT_REF_f'],
    normalize_axes=True,  # Data: use percentile scale (0-100)
    gridsize=11,  # Data: number of hexagons per side
    mincnt=5,  # Data: hide bins with fewer than 5 values
)
# The axis labels are chrome and move into FormatStyle. The colorbar params
# (zlabel, cmap, cb_*) are not chrome -- FormatStyle does not own the colorbar --
# so they stay as direct plot() arguments.
hm.plot(
    ax=None,  # Create new figure
    format_style=dv.plotting.FormatStyle(
        xlabel='Air temperature (percentile)',
        ylabel='Vapor pressure deficit (percentile)',
    ),
    cmap='RdYlBu_r',  # Colorbar: colormap name
    zlabel='NEE aggregated',  # Colorbar: label
    show_values=True,  # Show aggregated values on hexagons
    cb_digits_after_comma=1  # Colorbar: decimal places
)

plt.show(block=False)

print("Plotted hexbin with percentile normalization")
