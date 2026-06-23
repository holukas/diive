"""
==================================
Compound Extremes Quadrant Scatter
==================================

Quadrant scatter of two standardized drivers (z-scores), coloured and marked by
compound-extreme category with dashed threshold lines, using
:class:`~diive.core.plotting.compoundextremes.CompoundExtremesPlot` (after Wang et al., Fig. 2).

The classification itself comes from
:class:`~diive.analysis.compoundextremes.CompoundExtremes` (see
``examples/analysis/analysis_compound_extremes.py``); this example focuses on the plot.

This example covers:

- Building the plot directly from a fitted ``CompoundExtremes`` instance
- Custom category styling (colours / markers / legend labels)
- Building the plot from pre-classified data (no analysis class required)
- Chrome via ``FormatStyle``

Best for: visualizing which months/days fall into the air / soil / compound dry-hot
quadrants.
"""

import matplotlib.pyplot as plt

import diive as dv
from diive.plotting import CompoundExtremesPlot, FormatStyle

# %%
# Classify, then plot from the analysis instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``from_compound_extremes`` wires the z-score columns, categories, period labels,
# and signed threshold lines straight from the analysis instance.

df = dv.load_exampledata_parquet()
ce = dv.analysis.CompoundExtremes(
    var1=df['VPD_f'], var2=df['SWC_FF0_0.15_1'],
    agg='monthly', standardize_by='record',
    var1_extreme='high', var2_extreme='low', threshold=2.0,
    var1_label='Air', var2_label='Soil',
)

ce_plot = CompoundExtremesPlot.from_compound_extremes(ce)
fig, ax = plt.subplots(figsize=(9, 8))
ce_plot.plot(ax=ax, format_style=FormatStyle(title='Compound dry-hot extremes',
                                             xlabel='VPD z-score', ylabel='SWC z-score'))
# fig.show()  # disabled for the example gallery

# %%
# Custom category styling
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Override colours, markers, and legend labels per category. Keys are the fixed
# CompoundExtremes category codes: 'none', 'var1', 'var2', 'compound'.

styles = {
    'none': {'color': '#90A4AE', 'marker': '.', 'label': 'Normal'},
    'var1': {'color': '#FB8C00', 'marker': '^', 'label': 'Air dryness'},
    'var2': {'color': '#6D4C41', 'marker': 's', 'label': 'Soil dryness'},
    'compound': {'color': '#D32F2F', 'marker': 'D', 'label': 'Compound'},
}
ce_plot_styled = CompoundExtremesPlot(
    x=ce.results['VPD_f_Z'], y=ce.results['SWC_FF0_0.15_1_Z'],
    category=ce.results['CATEGORY'], category_styles=styles,
    labels=ce.results['PERIOD'], threshold_x=2.0, threshold_y=-2.0,
)
fig, ax = plt.subplots(figsize=(9, 8))
ce_plot_styled.plot(ax=ax, markersize=70, legend_title='Type')
# fig.show()

# %%
# Build from pre-classified data (no analysis class)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The plot is reusable on any pre-classified data: pass aligned x/y z-score Series
# and a category Series of arbitrary labels — styling falls back to a palette.

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.RangeIndex(200)
x = pd.Series(rng.normal(0, 1, 200), index=idx, name='driver-1 z')
y = pd.Series(rng.normal(0, 1, 200), index=idx, name='driver-2 z')
cat = pd.Series(np.where((x > 2) & (y < -2), 'both',
                np.where(x > 2, 'A', np.where(y < -2, 'B', 'normal'))),
                index=idx)

plot = CompoundExtremesPlot(x=x, y=y, category=cat,
                            category_order=['normal', 'A', 'B', 'both'],
                            threshold_x=2.0, threshold_y=-2.0)
fig, ax = plt.subplots(figsize=(9, 8))
plot.plot(ax=ax, annotate=False)
# fig.show()
