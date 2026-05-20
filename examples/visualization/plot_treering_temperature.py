"""
=================================
Tree-Ring Air Temperature Plot
=================================

Circular spiral plot showing daily air temperature as concentric annual rings.

Each ring represents one year (inner = older, outer = more recent). Colors encode
the daily mean temperature: blue for cold days, red for warm days. Months are
arranged around the circle with January at the bottom and July at the top, so the
seasonal cycle reads naturally.
"""

# %%
# Load data
# ^^^^^^^^^^
#
# Load the 30-minute example data directly.  The class handles resampling
# internally via the ``resample_freq`` parameter, so no manual pre-processing
# is needed.

import matplotlib.pyplot as plt

import diive as dv

df = dv.load_exampledata_parquet()

print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} half-hourly records")
print(f"Tair_f range: {df['Tair_f'].min():.1f} to {df['Tair_f'].max():.1f} deg C")

# %%
# Create tree-ring plot
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Data is prepared in __init__; all styling parameters go in plot().
# ``resample_freq='D'`` (default) aggregates observations to daily means before
# building the ring grid.  Use ``'h'`` or ``'30min'`` for finer angular resolution.

tr = dv.plot_treering(
    df=df,
    value_col='Tair_f',  # Data: column with values to visualize
    resample_freq='D'  # Data: resample to daily means (366 slots per ring)
)

# vmin/vmax are clipped relative to the data range (-23 to +28 deg C), so the
# colorbar automatically extends with arrows at both ends.
tr.plot(
    figsize=(10, 10),  # Styling: figure size
    cmap='RdBu_r',  # Styling: blue=cold, red=warm
    # vmin=-15,  # Styling: clips data below -23 -> downward arrow added
    # vmax=15,  # Styling: clips data above +28 -> upward arrow added
    title='Air temperature at ICOS Ecosystem Station Davos (2013-2022)',
    show_month_labels=True,  # Styling: month names around the ring
    show_month_lines=False,  # Styling: thin radial lines at month boundaries
    show_year_labels=True,  # Styling: year numbers on rings
    show_year_separators=True,  # Styling: thin circles between year rings
    year_label_frequency=1,  # Styling: label every year (only 10 years of data)
    cb_label='Air temperature (deg C)',
    cb_digits_after_comma=0
)

plt.show(block=False)

print("Plotted tree-ring air temperature visualization")
