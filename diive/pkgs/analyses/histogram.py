from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame


class Histogram:
    """
    Calculate histogram from Series in DataFrame

    kudos:
    - https://datagy.io/numpy-histogram/
    - https://www.adamsmith.haus/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
    """

    def __init__(self,
                 df: DataFrame,
                 col: str,
                 method: Literal['uniques'] = 'uniques',
                 ignore_fringe_bins: list = None):
        """
        Args:
            df: DataFrame
            col: Name of variable in *df*
            method: Method used for binning data
                Options:
                    - 'uniques': Each unique value in the dataset is a separate bin
            ignore_fringe_bins: List of integers [i, j] with length 2
                If a list is provided, then the first i and last j number of
                bins are removed from the results and ignored during
                distribution analyses.
                Example:
                    - Results with 'ignore_fringe_bins=None':
                         BIN_START_INCL     COUNTS
                            0.00            218
                            0.05            25
                            0.10            22
                            0.15            16
                            0.20            17
                             ...            ...
                            9.75            5
                            9.80            15
                            9.85            10
                            9.90            28
                            9.95            194
                    - Results with 'ignore_fringe_bins=[1, 5]':
                        BIN_START_INCL      COUNTS
                            0.05            25
                            0.10            22
                            0.15            16
                            0.20            17
                            0.25            12
                             ...            ...
                            9.50            7
                            9.55            9
                            9.60            4
                            9.65            6
                            9.70            16
        """
        self.method = method
        self.ignore_edge_bins = ignore_fringe_bins

        self.data = df[col].copy().dropna().to_numpy()
        self._results_df = None

        self._calc()

    @property
    def results(self) -> DataFrame:
        """Return bootstrap results for daily fluxes, includes threshold detection"""
        if not isinstance(self._results_df, DataFrame):
            raise Exception('Results for histogram calculation are empty')
        return self._results_df

    @property
    def peakbin(self):
        # Find bin with maximum counts
        if not isinstance(self._results_df, DataFrame):
            raise Exception('Results for histogram calculation are empty')
        idx_maxcounts = self._results_df['COUNTS'].idxmax()
        return self._results_df['BIN_START_INCL'].iloc[idx_maxcounts]

    def _calc(self):
        # Set binning method
        bins = self._binning_method()

        # Count values per bin
        counts, bin_edges = np.histogram(self.data, bins=bins, range=None, weights=None, density=False)

        # Remove fringe bins
        if self.ignore_edge_bins:
            _start = self.ignore_edge_bins[0]
            _end = self.ignore_edge_bins[1]
            counts = counts[_start:-_end]
            bin_edges = bin_edges[_start:-(_end + 1)]
        else:
            # Make same length for collection in DataFrame
            bin_edges = bin_edges[0:-1]

        # Collect in dataframe
        self._results_df = pd.DataFrame()
        self._results_df['BIN_START_INCL'] = bin_edges
        self._results_df['COUNTS'] = counts

    def _binning_method(self):
        bins = None
        if self.method == 'uniques':
            # Use unique values as bins, each unique value is a separate bin
            bins = np.unique(self.data)
        return bins

        # df.plot.bar(x='BIN_START_INCL', y='COUNTS')
        # plt.show()


def example():
    from diive.core.io.filereader import ReadFileType
    SOURCE = r"F:\01-NEW\IRGACOMP\DAV\NON-ICOS\2013_1__IRGA7572\fluxrun\IRGA72_CH-DAV_FR-20221215-171820_OPENLAG\2-0_eddypro_flux_calculations\results\IRGA72_2013_1_eddypro_CH-DAV_FR-20221215-171820_full_output_2022-12-15T211317_adv_OPENLAG.csv"
    loaddatafile = ReadFileType(filetype='EDDYPRO_FULL_OUTPUT_30MIN', filepath=SOURCE, data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()

    df = data_df.copy()
    col = 'co2_time_lag'

    h = Histogram(df=df, col=col, method='uniques', ignore_fringe_bins=[1, 5])
    print(h.results)
    print(h.peakbin)

    # bins = 10
    # valuebins = True
    # binsize = 0.05
    # bins = np.arange(data.min(), data.max(), binsize)


if __name__ == '__main__':
    example()
