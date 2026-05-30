"""
ANALYSIS: HISTOGRAM DISTRIBUTION
=================================

Distribution analysis with flexible binning methods and statistical summaries.
Supports fringe bin removal and detailed bin-wise statistics.

Part of the diive library: https://github.com/holukas/diive
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class Histogram:

    def __init__(self,
                 series: Series = None,
                 method: Literal['n_bins', 'uniques'] = 'n_bins',
                 n_bins: int = 10,
                 ignore_fringe_bins: list = None,
                 s: Series = None):
        """Calculate histogram from Series.

        Args:
            series: A pandas Series.
            method: Method used for binning data
                Options:
                    - 'uniques': Each unique value in the dataset is a separate bin
                    - 'n_bins': Number of bins
            n_bins: Number of bins, needed if *method* is 'n_bins', otherwise ignored.
            ignore_fringe_bins: List of integers [i, j] with length 2
                If a list is provided, then the first i and last j number of
                bins are removed from the results and ignored during
                distribution analysis.
            s: Deprecated alias for *series*.

        Properties:
            .results: Histogram results as DataFrame with BIN_START_INCL and COUNTS columns
            .peakbins: Top 5 bins by count

        Example:
            See `examples/analysis/analysis_histogram_distribution.py` for complete examples.

        See Also:
            Histogram : This class.
        """
        # `s` is the deprecated name for `series` (renamed for consistency with
        # the other diive classes, which all take `series`).
        if s is not None:
            warnings.warn("Histogram: the `s` argument is deprecated, use `series` instead.",
                          DeprecationWarning, stacklevel=2)
            series = s if series is None else series
        if series is None:
            raise ValueError("Histogram requires `series`.")

        self.method = method
        self.n_bins = n_bins
        self.ignore_edge_bins = ignore_fringe_bins

        self.series = series.copy().dropna().to_numpy()
        self._results_df = None

        self._calc()

    @property
    def results(self) -> DataFrame:
        """Return bootstrap results for daily fluxes, includes threshold detection"""
        if not isinstance(self._results_df, DataFrame):
            raise Exception('Results for histogram calculation are empty')
        return self._results_df

    @property
    def peakbins(self):
        """Returns the five bins with the most counts in decreasing order"""
        ix_maxcounts = self.results['COUNTS'].sort_values(ascending=False).head(5).index
        peakbins = self.results['BIN_START_INCL'].iloc[ix_maxcounts]
        return peakbins.to_numpy().tolist()
        # Find bin with maximum counts
        # idx_maxcounts = self.results['COUNTS'].idxmax()
        # return self.results['BIN_START_INCL'].iloc[idx_maxcounts]

    def _calc(self):
        # Set binning method
        bins = self._binning_method()

        # Count values per bin
        counts, bin_edges = np.histogram(self.series, bins=bins, range=None, weights=None, density=False)

        # Remove fringe bins
        if self.ignore_edge_bins:
            counts, bin_edges = self._ignore_fringe_bins(counts=counts, bin_edges=bin_edges)
        else:
            # Make same length for collection in DataFrame
            bin_edges = bin_edges[0:-1]

        # Collect in dataframe
        self._results_df = pd.DataFrame()
        self._results_df['BIN_START_INCL'] = bin_edges
        self._results_df['COUNTS'] = counts

    def _ignore_fringe_bins(self, counts, bin_edges):
        _start = self.ignore_edge_bins[0]
        _end = self.ignore_edge_bins[1]
        # Use len()-based stops so that _end == 0 trims nothing from the end.
        # A naive counts[_start:-_end] would be counts[_start:0] (empty) when _end is 0.
        counts = counts[_start:len(counts) - _end]
        bin_edges = bin_edges[_start:len(bin_edges) - _end - 1]
        return counts, bin_edges

    def _binning_method(self):
        bins = None
        if self.method == 'uniques':
            # Use unique values as bins, each unique value is a separate bin
            bins = np.unique(self.series)
        elif self.method == 'n_bins':
            # Input defines number of bins
            bins = self.n_bins
        return bins

        # df.plot.bar(x='BIN_START_INCL', y='COUNTS')
        # plt.show()
