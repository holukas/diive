import unittest

from diive.configs.exampledata import load_exampledata_parquet
from diive.pkgs.createvar.timesince import TimeSince


class TestCreateVar(unittest.TestCase):

    def test_timesince(self):
        df = load_exampledata_parquet()
        series_ta = df.loc[(df.index.year == 2022) & (df.index.month == 3), "Tair_f"].copy()
        ts = TimeSince(series_ta, upper_lim=5, lower_lim=None, include_lim=True)
        ts.calc()
        ts_full_results = ts.get_full_results()
        greater_equal_stats = ts_full_results.loc[ts_full_results['Tair_f'] >= 5].describe()
        less_stats = ts_full_results.loc[ts_full_results['Tair_f'] < 5].describe()
        self.assertEqual(greater_equal_stats['Tair_f']['count'], 273)
        self.assertEqual(greater_equal_stats['Tair_f']['min'], 5.017)
        self.assertEqual(ts_full_results['FLAG_IS_OUTSIDE_RANGE'].sum(), 273)
        self.assertEqual(less_stats['Tair_f']['count'], 1215)
        self.assertEqual(less_stats['Tair_f']['max'], 4.99)
        self.assertEqual(less_stats['FLAG_IS_OUTSIDE_RANGE']['min'], 0)
        self.assertEqual(less_stats['FLAG_IS_OUTSIDE_RANGE']['max'], 0)
        self.assertEqual(ts_full_results.sum().sum(), -7223.621999999999)
        # from pathlib import Path
        # outpath = Path(r"F:\TMP") / 'ts_full_results.csv'
        # ts_full_results.to_csv(outpath, index=False)
        # ts_series = ts.get_timesince()


if __name__ == '__main__':
    unittest.main()
