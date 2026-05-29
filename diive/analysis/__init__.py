"""
ANALYSIS: TIME SERIES ANALYSIS METHODS
=======================================

Comprehensive toolkit: decomposition, correlation, gap detection, histograms, grid aggregation.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.analysis.correlation import DailyCorrelation
from diive.analysis.granger import GrangerCausality

# Alias for backward compatibility
daily_correlation = DailyCorrelation

from diive.analysis.decoupling import StratifiedAnalysis
from diive.analysis.gapfinder import GapFinder, GapStats
from diive.analysis.gridaggregator import GridAggregator
from diive.analysis.harmonic import harmonic_analysis
from diive.analysis.histogram import Histogram
from diive.analysis.optimumrange import FindOptimumRange
from diive.analysis.quantiles import percentiles101
from diive.analysis.seasonaltrend import SeasonalTrendDecomposition
from diive.fits.fitter import BinFitterCP

__all__ = [
    'BinFitterCP',
    'DailyCorrelation',
    'daily_correlation',
    'GrangerCausality',
    'StratifiedAnalysis',
    'GapFinder',
    'GapStats',
    'GridAggregator',
    'harmonic_analysis',
    'Histogram',
    'FindOptimumRange',
    'percentiles101',
    'SeasonalTrendDecomposition',
]
