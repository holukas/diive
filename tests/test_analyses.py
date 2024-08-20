import unittest

from diive.pkgs.analyses.histogram import Histogram


class TestCreateVar(unittest.TestCase):

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
        self.assertEqual(hist.peakbins, [1.148, 1.241, 1.929, 0.765, 1.632])


if __name__ == '__main__':
    unittest.main()
