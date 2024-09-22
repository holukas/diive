import unittest


class TestCorrections(unittest.TestCase):

    def test_winddiroffset(self):
        from diive.configs.exampledata import load_exampledata_winddir
        from diive.pkgs.corrections.winddiroffset import WindDirOffset
        df = load_exampledata_winddir()
        # Get wind direction time series as series
        winddir = df['wind_dir'].copy()
        locs = (winddir.index.year >= 2020) & (winddir.index.year <= 2022)
        winddir = winddir.loc[locs]
        winddir = winddir.dropna()
        wds = WindDirOffset(winddir=winddir, offset_start=-50, offset_end=50,
                            hist_ref_years=[2021, 2022], hist_n_bins=360)
        yearlyoffsets_df = wds.get_yearly_offsets()
        winddir_corrected = wds.get_corrected_wind_directions()
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2020, 'OFFSET'].values, -2)
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2021, 'OFFSET'].values, 0)
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2022, 'OFFSET'].values, 0)
        self.assertEqual(winddir_corrected.sum(), 7495054.8)
        self.assertEqual(winddir_corrected.max(), 359.9)
        self.assertEqual(winddir_corrected.min(), 0)


if __name__ == '__main__':
    unittest.main()
