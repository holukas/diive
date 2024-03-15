"""

=============
WINDDIROFFSET
=============

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.pkgs.analyses.histogram import Histogram


class WindDirOffset:
    """
    Compare yearly wind direction histograms to reference, detect
    offset in comparison to reference and correct wind directions
    for offset per year

    - Example notebook available in:
        notebooks/Corrections/WindDirectionOffset.ipynb
    """

    def __init__(self,
                 winddir: Series,
                 hist_ref_years: list,
                 offset_start: int = -100,
                 offset_end: int = 100,
                 hist_n_bins: int = 360):
        """
        Build histogram of wind directions for each year and compare to reference
        histogram built from data in reference years

        (1) Build reference histogram of wind directions from reference years
        (2) For each year:
            2a: Add constant offset to wind directions, starting with *offset_start*
            2b: Build histogram of wind directions
            2c: Calculate absolute correlation between 2b and reference
            2d: Continue with next offset, ending with *offset_end*
            2e: Detect offset that yielded maximum absolute correlation with reference

        Args:
            winddir: Time series of wind directions in degrees
            hist_ref_years: List of years for building reference histogram, e.g. "[2021, 2022]"
            offset_start: Minimum offset in degrees to shift *s*
            offset_end: Maximum offset in degrees to shift *s*
            hist_n_bins: Number of bins for building histograms
        """
        self.winddir = winddir
        self.hist_ref_years = hist_ref_years
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.hist_n_bins = hist_n_bins

        # Wind directions shifted by offset that yielded maximum absolute
        # correlation with reference
        self.winddir_shifted = self.winddir.copy()

        # Unique years
        self.uniq_years = list(self.winddir.index.year.unique())

        # Reference histogram
        self.ref_results = self._reference_histogram()

        # Find offset per year and store results in dict
        self.shiftdict = self._calc_histogram_correlations()

        # Detect offset per year
        self.yearlyoffsets_df = self._find_yearly_offsets()

        # Correct wind directions
        self._correct_wind_directions()

        # self.showplots()

    def get_corrected_wind_directions(self):
        return self.winddir_shifted

    def get_yearly_offsets(self) -> DataFrame:
        """Return yearly wind direction offsets that yielded maximum absolute
        correlation with reference"""
        return self.yearlyoffsets_df

    def _correct_wind_directions(self):
        """Correct wind directions by yearly offsets"""
        for year in self.uniq_years:
            offset = int(self.yearlyoffsets_df.loc[self.yearlyoffsets_df['YEAR'] == year]['OFFSET'].values)
            self.winddir_shifted.loc[self.winddir_shifted.index.year == year] += offset
        self.winddir_shifted = self._correct_degrees(self.winddir_shifted)

    def _find_yearly_offsets(self):
        yearlyoffsets_df = pd.DataFrame(columns=['YEAR', 'OFFSET'])
        for key, val in self.shiftdict.items():
            val_sorted = val.sort_values(by='CORR_ABS', ascending=False).copy()
            shift_maxcorr = val_sorted.iloc[0]['SHIFT']
            yearlyoffsets_df.loc[len(yearlyoffsets_df)] = [key, shift_maxcorr]
        return yearlyoffsets_df

    def _calc_histogram_correlations(self):
        """ """
        shiftdict = {}
        for year in self.uniq_years:
            print(f"Working on year {year} ...")
            s_year = self.winddir.loc[self.winddir.index.year == year].copy()
            shiftdf_year = pd.DataFrame(columns=['SHIFT', 'CORR_ABS'])
            for shift in np.arange(self.offset_start, self.offset_end, 1):
                s_year_shifted = s_year.copy()
                s_year_shifted += shift
                s_year_shifted = self._correct_degrees(s=s_year_shifted)
                histo_shifted = Histogram(s=s_year_shifted, method='n_bins', n_bins=360)
                results_shifted = histo_shifted.results
                corr_abs = abs(results_shifted['COUNTS'].corr(self.ref_results['COUNTS']))
                shiftdf_year.loc[len(shiftdf_year)] = [shift, corr_abs]
            shiftdict[year] = shiftdf_year
        return shiftdict

    def showplots(self):
        """Plot absolute correlations for each year"""
        for key, val in self.shiftdict.items():
            shiftdf = val.set_index(keys='SHIFT', drop=True)
            shiftdf.plot()
            plt.show()

    def _correct_degrees(self, s: Series):
        """Correct degree values that go below zero or above 360"""
        _locs_above360 = s > 360
        s[_locs_above360] -= 360
        _locs_belowzero = s < 0
        s[_locs_belowzero] += 360
        return s

    def _reference_histogram(self):
        """Calculate reference histogram"""
        select_years = self.winddir.index.year.isin(self.hist_ref_years)
        ref_s = self.winddir[select_years]
        ref_histo = Histogram(s=ref_s, method='n_bins', n_bins=self.hist_n_bins)
        ref_results = ref_histo.results
        return ref_results


def example():
    # # Load example data
    # from diive.core.io.filereader import ReadFileType
    # SOURCE = r"F:\01-NEW\FF202303\FRU\Level-0_OPENLAG_results_2005-2022\OUT_DIIVE-20230410-020904\winddir_Dataset_DIIVE-20230410-020904_Original-30T.diive.csv"
    # loaddatafile = ReadFileType(filetype='DIIVE-CSV-30MIN', filepath=SOURCE, data_nrows=None)
    # data_df, metadata_df = loaddatafile.get_filedata()

    # # Save data as pickle for faster loading
    # from diive.core.io.files import load_pickle, save_as_pickle
    # pickle_ = save_as_pickle(data=data_df, outpath='F:\_temp', filename='temp')

    # Load data from pickle for faster loading
    from diive.core.io.files import load_pickle
    data_df = load_pickle(filepath=r"L:\Sync\luhk_work\_temp\temp.pickle")

    # Get wind direction time series as series
    col = 'wind_dir'
    s = data_df[col].copy()

    # Prepare input data
    s = s.loc[s.index.year <= 2022]
    s = s.dropna()

    wds = WindDirOffset(winddir=s, offset_start=-50, offset_end=50, hist_ref_years=[2021, 2022], hist_n_bins=360)
    yearlyoffsets_df = wds.get_yearly_offsets()
    s_corrected = wds.get_corrected_wind_directions()
    print(yearlyoffsets_df)
    print(s_corrected)
    print(s)

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=s_corrected).show()
    HeatmapDateTime(series=s).show()


if __name__ == '__main__':
    example()
