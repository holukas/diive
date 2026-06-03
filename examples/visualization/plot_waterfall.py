"""
=========================
Waterfall Flux Budget Plot
=========================

Financial-style waterfall chart of daily CO2 contributions building up a running
flux budget. Each day floats from the previous running total, colored by uptake
(sink) vs. release (source).

Best for: Daily CO2 uptake/release budgets, visualizing how a seasonal or annual
net flux accumulates day by day.
"""

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df_orig = dv.load_exampledata_parquet()

# Convert units: umol CO2 m-2 s-1 --> g C m-2 30min-1
conversion_factor = 0.02161926
series_units = r'($\mathrm{gC\ m^{-2}}$)'

series = df_orig['NEE_CUT_REF_f'].multiply(conversion_factor)
series = series.loc[(series.index.year == 2019) & (series.index.month == 6)]

# %%
# Daily CO2 waterfall
# ^^^^^^^^^^^^^^^^^^^
#
# Half-hourly NEE is aggregated to daily sums internally. With the NEE convention
# (negative = uptake), uptake days are blue and release days are red. The running
# total builds toward the net annual flux annotated at the end.

dv.plotting.WaterfallPlot(
    series=series,
    series_units=series_units,
    resample='D',
    uptake_is_negative=True,
).plot(showplot=True)
