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
    """
    Calculate histogram from Series in DataFrame
    """

    def __init__(self,
                 s: Series,
                 method: Literal['n_bins', 'uniques'] = 'n_bins',
                 n_bins: int = 10,
                 ignore_fringe_bins: list = None):
        """
        Args:
            s: Time series
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
        """Returns the five bins with the most counts"""
        ix_maxcounts = self.results['COUNTS'].sort_values(ascending=False).head(5).index
        peakbins = self.results['BIN_START_INCL'].iloc[ix_maxcounts]
        return list(peakbins.values)
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


def example():
    # # from diive.core.io.filereader import ReadFileType
    # # SOURCE = r"F:\01-NEW\FF202303\FRU\Level-0_OPENLAG_results_2005-2022\OUT_DIIVE-20230410-020904\winddir_Dataset_DIIVE-20230410-020904_Original-30T.diive.csv"
    # # loaddatafile = ReadFileType(filetype='DIIVE-CSV-30MIN', filepath=SOURCE, data_nrows=None)
    # # data_df, metadata_df = loaddatafile.get_filedata()
    #
    # # from diive.core.io.files import load_pickle, save_as_pickle
    # # pickle_ = save_as_pickle(data=data_df, outpath='F:\_temp', filename='temp')
    #
    # from diive.core.io.files import load_pickle
    # data_df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\temp.pickle")
    #
    # col = 'wind_dir'
    # s = data_df[col].copy()
    #
    # s = s.loc[s.index.year <= 2022]
    # # s = s.loc[s.index.year <= 2022]
    # # s = s.loc[(s.index.month >= 8) & (s.index.month <= 8)]
    # s = s.dropna()

    # # Wind direction correction for certain years
    # _locs = (s.index.year >= 2005) & (s.index.year <= 2019)
    # s[_locs] -= 25
    # _locs_below_zero = s < 0
    # s[_locs_below_zero] += 360

    # s = s.astype(int)
    # rounded = round(s / 10) * 10

    # # Calculate wind direction e.g. per YEAR-MONTH, e.g. 2022-06, 2022-07, ...
    # sdf = pd.DataFrame(s)
    # # sdf['YEAR-MONTH'] = sdf.index.year.astype(str).str.cat(sdf.index.month.astype(str).str.zfill(2), sep='-')
    # # grouped = sdf.groupby(sdf['YEAR-MONTH'])
    # sdf['YEAR'] = sdf.index.year.astype(str)
    # grouped = sdf.groupby(sdf['YEAR'])
    # wdavg = pd.Series()
    # wdp25 = pd.Series()
    # wdp75 = pd.Series()
    # for g, d in grouped:
    #     wd_avg = direction_avg_kanda(angles=d['wind_dir'], agg='median')
    #     wdavg.loc[g] = wd_avg
    #     wd_p25 = direction_avg_kanda(angles=d['wind_dir'], agg='P25')
    #     wdp25.loc[g] = wd_p25
    #     wd_p75 = direction_avg_kanda(angles=d['wind_dir'], agg='P75')
    #     wdp75.loc[g] = wd_p75
    # import matplotlib.pyplot as plt
    # wdavg.plot()
    # wdp25.plot()
    # wdp75.plot()
    # plt.show()
    # plt.show()

    # gr = sdf.groupby(sdf['YEAR-MONTH']).apply(direction_avg_kanda)

    # s.apply(direction_avg_kanda)
    # winddir_avg = direction_avg_kanda(angles=_s.to_numpy())

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=s).show()

    # # Reference histogram from 2021 and 2022
    # ref_s = s.loc[s.index.year >= 2021]
    # ref_histo = Histogram(s=ref_s, method='n_bins', n_bins=360)
    # ref_results = ref_histo.results

    # # Test year
    # test_s = s.loc[s.index.year == 2020]
    # shiftdf = pd.DataFrame(columns=['SHIFT', 'CORR'])
    # for shift in np.arange(-100, 100, 1):
    #     # print(shift)
    #     test_s_shifted = test_s.copy()
    #     test_s_shifted += shift
    #
    #     _locs_above360 = test_s_shifted > 360
    #     test_s_shifted[_locs_above360] -= 360
    #     _locs_belowzero = test_s_shifted < 0
    #     test_s_shifted[_locs_belowzero] += 360
    #
    #     test_histo = Histogram(s=test_s_shifted, method='n_bins', n_bins=360)
    #     test_results = test_histo.results
    #     corr = test_results['COUNTS'].corr(ref_results['COUNTS'])
    #     shiftdf.loc[len(shiftdf)] = [shift, corr]
    #     # print(f"{shift:.1f}: corr = {corr}")
    #
    # import matplotlib.pyplot as plt
    # shiftdf = shiftdf.set_index(keys='SHIFT', drop=True)
    # shiftdf.plot()
    # plt.show()
    # print("X")

    shiftdf = shiftdf.sort_values(by='CORR', ascending=False).copy()
    shift_maxcorr = shiftdf.iloc[0].name
    print(shift_maxcorr)

    # g = s.groupby(s.index.year)
    # for year, _s in g:
    #     h = Histogram(s=_s, method='n_bins', n_bins=36)
    #     # h = Histogram(s=s, method='uniques', ignore_fringe_bins=[1, 5])
    #     print(h.results)
    #     print(f"{year}: {h.peakbins}")

    # # bins = 10
    # # valuebins = True
    # # binsize = 0.05
    # # bins = np.arange(data.min(), data.max(), binsize)


if __name__ == '__main__':
    example()
