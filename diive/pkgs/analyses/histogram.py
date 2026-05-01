"""
kudos:
- https://datagy.io/numpy-histogram/
- https://www.adamsmith.haus/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
"""

from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class Histogram:

    def __init__(self,
                 s: Series,
                 method: Literal['n_bins', 'uniques'] = 'n_bins',
                 n_bins: int = 10,
                 ignore_fringe_bins: list = None):
        """Calculate histogram from Series.

        Args:
            s: A pandas Series.
            method: Method used for binning data
                Options:
                    - 'uniques': Each unique value in the dataset is a separate bin
                    - 'n_bins': Number of bins
            n_bins: Number of bins, needed if *method* is 'n_bins', otherwise ignored.
            ignore_fringe_bins: List of integers [i, j] with length 2
                If a list is provided, then the first i and last j number of
                bins are removed from the results and ignored during
                distribution analyses.

        Properties:
            .results: Histogram results as DataFrame with BIN_START_INCL and COUNTS columns
            .peakbins: Top 5 bins by count

        Example:
            See `examples/analyses/histogram.py` for complete examples.
        """
        self.method = method
        self.n_bins = n_bins
        self.ignore_edge_bins = ignore_fringe_bins

        self.series = s.copy().dropna().to_numpy()
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
        return peakbins.values.tolist()
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
        counts = counts[_start:-_end]
        bin_edges = bin_edges[_start:-(_end + 1)]
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
