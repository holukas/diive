"""
========================
Shifted Distribution Plot
========================

Compare how a variable's distribution has shifted between a reference and a
comparison period. Zone boundaries are derived automatically from the reference
period's mean and standard deviation (Hansen et al. methodology: +-1sigma and
+-3sigma).

Best for: Detecting long-term shifts in climate variables, comparing two decades.
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()
series.name = 'Air Temperature (deg C)'

print(f"Loaded {len(series)} records from {series.index[0]} to {series.index[-1]}")
print(f"Mean: {series.mean():.2f} deg C, Std: {series.std():.2f} deg C")

# %%
# Create shifted distribution plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Reference period: 2013-2016 (first 4 years)
# Comparison period: 2019-2022 (last 4 years)
#
# The gray hatched curve shows the reference distribution.
# The colored filled curve shows the comparison distribution, with zones
# colored from blue (cold) to red (hot).

sdp = dv.plotting.ShiftedDistributionPlot(
    series=series,
    ref_period=('2013', '2016'),
    comp_period=('2019', '2022'),
)

print(f"\nBreakpoints (from reference period):")
labels = ['Extremely cold / Cold', 'Cold / Normal', 'Normal / Hot', 'Hot / Extremely hot']
for label, bp in zip(labels, sdp.breakpoints):
    print(f"  {label}: {bp:.2f} deg C")

sdp.plot(
    ax=None,
    title='Air Temperature Distribution Shift',
    xlabel='Air Temperature (deg C)',
    ref_label='Reference 2013-2016',
    comp_label='Comparison 2019-2022',
    show_legend=True,
    show_title=True,
    show_xaxis=True,   # set False to hide bottom spine, ticks, and tick labels
    show_yaxis=False,   # set False to hide left spine, ticks, and tick labels
    figsize=(16, 7),
)

