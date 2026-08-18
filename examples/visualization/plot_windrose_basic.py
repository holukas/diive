"""
=========================================
Wind Rose: Variable by Wind Direction
=========================================

Radial plot that aggregates a variable into wind-direction sectors. Each compass
sector is drawn as a polar bar whose length is the sector's aggregated value
(mean, median, min, max, sum, std or count), with North at the top and angles
increasing clockwise — the standard meteorological wind-rose layout.

Here we look at the directional response of the CO2 flux: which wind sectors the
station sees net uptake (negative flux) from, and which it sees release from.
"""

# %%
# Load data
# ^^^^^^^^^
#
# The EddyPro full-output example carries ``wind_dir`` (degrees) alongside the
# fluxes and meteo variables in a single dataframe, so no merging is needed.

import matplotlib.pyplot as plt

import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN

df, meta = load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()

print(f"Data: {len(df)} records, wind_dir {df['wind_dir'].min():.0f}-{df['wind_dir'].max():.0f} deg")

# %%
# Aggregate and plot
# ^^^^^^^^^^^^^^^^^^^
#
# Data + aggregation parameters go to ``__init__`` (Phase 1); all styling lives in
# ``plot()`` (Phase 2). ``agg='mean'`` reduces the CO2 flux to a per-sector mean;
# bars are coloured by value with a diverging colormap, so blue sectors (net
# uptake) and red sectors (net release) read at a glance. The per-sector table is
# always available on ``.results`` and is printed as a Rich table with
# ``verbose=True``.

rose = dv.plotting.WindRosePlot(
    series=df['co2_flux'],  # Data: variable to aggregate
    wind_dir=df['wind_dir'],  # Data: wind direction in degrees
    agg='mean',  # Data: per-sector aggregation
    n_sectors=64,  # Data: 16 compass sectors (N, NNE, NE, ...)
    verbose=True,  # prints the Rich per-sector report
)

style = dv.plotting.FormatStyle(title='Mean CO2 flux by wind direction (CH-CHA)')
rose.plot(
    format_style=style,
    cmap='RdBu_r',  # Styling: blue = uptake, red = release
    cb_label='Mean CO2 flux',
    # cb_digits_after_comma defaults to None -> integer colorbar labels here,
    # since the flux ticks (-25, -20, ... 5) need no decimals.
)

plt.show(block=False)

# %%
# Colour by a second variable
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Pass ``z`` to colour the bars by a *different* variable than the one driving
# the bar lengths. Here the bar length is still the mean CO2 flux, but the colour
# now encodes the mean air temperature per sector (aggregated by ``z_agg``). The
# bar value goes to ``.results`` as usual; the colour aggregate is added as ``Z``.

rose_z = dv.plotting.WindRosePlot(
    series=df['co2_flux'],  # Data: bar length = mean CO2 flux
    wind_dir=df['wind_dir'],
    agg='mean',
    n_sectors=360,
    z=df['air_temperature'],  # Data: bar colour = mean air temperature
    z_agg='mean',
)
rose_z.plot(
    format_style=dv.plotting.FormatStyle(title='CO2 flux (length) coloured by air temperature'),
    cmap='plasma',
)

plt.show(block=False)

# %%
# Inspect the result table
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``.results`` is a tidy per-sector DataFrame holding every statistic, not just
# the plotted aggregate — handy for further analysis or export.

print(rose.results.round(2).to_string())
