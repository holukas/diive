"""
PLOTTING: PUBLIC NAMESPACE FOR THE PLOT CLASSES
===============================================

Public namespace ``dv.plotting``: the plot classes and ``FormatStyle``.

The plot classes follow a two-phase pattern: data and computation parameters go
to ``__init__``, styling goes to ``plot(ax=None, ...)``. Title, labels, font
sizes, grid and legend are set with a ``FormatStyle`` passed as ``format_style``
(polar plots use only some of its fields, see their ``plot()``).

Part of the diive library: https://github.com/holukas/diive
"""
from diive.core.plotting.bar import LongtermAnomaliesYear
from diive.core.plotting.compoundextremes import CompoundExtremesPlot
from diive.core.plotting.styles.format import FormatStyle
from diive.core.plotting.cumulative import Cumulative
from diive.core.plotting.cumulative import CumulativeYear
from diive.core.plotting.dielcycle import DielCycle
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.plotting.heatmap_datetime import HeatmapYearMonth
from diive.core.plotting.heatmap_xyz import HeatmapXYZ
from diive.core.plotting.hexbin import HexbinPlot
from diive.core.plotting.histogram import HistogramPlot
from diive.core.plotting.ridgeline import RidgeLinePlot
from diive.core.plotting.shifted_distribution import ShiftedDistributionPlot
from diive.core.plotting.scatter import ScatterXY
from diive.core.plotting.surface_grid import DateTimeSurface, datetime_surface_grid
from diive.core.plotting.timeseries import TimeSeries
from diive.core.plotting.treering import TreeRingPlot
from diive.core.plotting.waterfall import WaterfallPlot
from diive.core.plotting.windrose import WindRosePlot

__all__ = [
    'FormatStyle',
    'CompoundExtremesPlot',
    'LongtermAnomaliesYear',
    'Cumulative',
    'CumulativeYear',
    'DielCycle',
    'HeatmapDateTime',
    'HeatmapYearMonth',
    'HeatmapXYZ',
    'HexbinPlot',
    'HistogramPlot',
    'RidgeLinePlot',
    'ShiftedDistributionPlot',
    'ScatterXY',
    'DateTimeSurface',
    'datetime_surface_grid',
    'TimeSeries',
    'TreeRingPlot',
    'WaterfallPlot',
    'WindRosePlot',
]
