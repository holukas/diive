import unittest

from diive.pkgs.analyses.histogram import Histogram


class TestAnalyses(unittest.TestCase):

    def test_daily_correlation(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.analyses.correlation import daily_correlation
        from diive.pkgs.createvar.potentialradiation import potrad
        data_df = load_exampledata_parquet()
        # Use only year 2022
        data_df = data_df.loc[data_df.index.year == 2022].copy()
        # Observed radiation time series for example (1) and (2)
        rg_series = data_df['Rg_f'].copy()
        # Observed air temperature time series for example (2)
        ta_series = data_df['Tair_f'].copy()
        # Observed net ecosystem exhange of CO2 time series from eddy covariance measurements for example (3)
        nee_series = data_df['NEE_CUT_REF_f'].copy()
        # Calculate potential radiation SW_IN_POT to use as reference
        reference = potrad(timestamp_index=rg_series.index,
                           lat=47.286417,
                           lon=7.733750,
                           utc_offset=1)
        # Calculate daily correlation between Rg_f and SW_IN_POT
        daycorrs = daily_correlation(
            s1=rg_series,
            s2=reference,
            mincorr=0.8,
            showplot=False
        )
        self.assertEqual(daycorrs.sum(), 337.3189145385522)
        # Calculate daily correlation between Tair_f and NEE_CUT_REF_f
        daycorrs = daily_correlation(
            s1=ta_series,
            s2=nee_series,
            showplot=False
        )
        self.assertEqual(daycorrs.sum(), -167.25042524807637)
        self.assertEqual(daycorrs.min(), -0.9450031804629302)
        self.assertEqual(daycorrs.max(), 0.7109706199504967)

    def test_quantilexyaggz(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.analyses.quantilexyaggz import QuantileXYAggZ
        df = load_exampledata_parquet()
        # Make subset of three required columns
        vpd_col = 'VPD_f'
        ta_col = 'Tair_f'
        swin_col = 'Rg_f'
        df = df[[vpd_col, ta_col, swin_col]].copy()
        # Use data May and Sep
        df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()
        # Use daytime data
        daytime_locs = (df[swin_col] > 0)
        df = df[daytime_locs].copy()
        df = df.dropna()
        q = QuantileXYAggZ(
            x=df[swin_col],
            y=df[ta_col],
            z=df[vpd_col],
            n_quantiles=10,
            min_n_vals_per_bin=3,
            binagg_z='mean'
        )
        q.run()
        pivotdf = q.pivotdf.copy()
        self.assertEqual(pivotdf.sum().sum(), 573.6496756683449)
        self.assertEqual(len(pivotdf.columns), 10)
        self.assertEqual(len(pivotdf.index), 10)

    def test_histogram(self):
        from diive.configs.exampledata import load_exampledata_parquet
        data_df = load_exampledata_parquet()
        series = data_df['NEE_CUT_REF_f'].copy()

        hist = Histogram(s=series, method='n_bins', n_bins=10, ignore_fringe_bins=None)
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -40.811)
        self.assertEqual(bin_starts.mean(), -11.065549999999998)
        self.assertEqual(bin_starts.count(), 10)
        checkix = results.index[results['BIN_START_INCL'] == -1.1503999999999976].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 112210)
        self.assertEqual(hist.peakbins, [-1.1503999999999976, -7.7605, -14.3706, 5.459699999999998, -20.9807])

        hist = Histogram(s=series, method='n_bins', n_bins=10, ignore_fringe_bins=[1, 3])
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -34.2009)
        self.assertEqual(bin_starts.iloc[-1], -1.1503999999999976)
        self.assertEqual(bin_starts.mean(), -17.67565)
        self.assertEqual(bin_starts.count(), 6)
        checkix = results.index[results['BIN_START_INCL'] == -1.1503999999999976].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 112210)
        self.assertEqual(hist.peakbins, [-1.1503999999999976, -7.7605, -14.3706, -20.9807, -27.5908])

        hist = Histogram(s=series, method='uniques', n_bins=10, ignore_fringe_bins=[1, 3])
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -39.817)
        self.assertEqual(bin_starts.iloc[-1], 22.276)
        self.assertEqual(bin_starts.mean(), -3.605244071740962)
        self.assertEqual(bin_starts.count(), 24923)
        checkix = results.index[results['BIN_START_INCL'] == -8.062].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 7)
        self.assertEqual(hist.peakbins, [1.148, 1.241, 1.929, 1.324, 1.632])


if __name__ == '__main__':
    unittest.main()
