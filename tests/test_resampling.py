import unittest
import diive as dv
import diive.configs.exampledata as ed
from diive.core.times.resampling import diel_cycle
from diive.configs.exampledata import load_exampledata_parquet_long

class TestResampling(unittest.TestCase):

    def test_resample_to_monthly_agg_matrix(self):
        df = load_exampledata_parquet_long()
        series = df['Tair_f'].copy()
        monthly_means = dv.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=False)
        monthly_means_ranks = dv.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=True)
        self.assertEqual(monthly_means.shape, (26, 12))
        self.assertEqual(monthly_means.loc[1997, 3], 1.083760752688172)
        self.assertEqual(monthly_means.loc[2019, 6], 14.38481597222222)
        self.assertEqual(monthly_means_ranks.loc[1997, 3], 5)
        self.assertEqual(monthly_means_ranks.loc[2019, 6], 2)
        self.assertEqual(monthly_means.sum().sum(), 1335.6155073307543)
        self.assertEqual(monthly_means_ranks.sum().sum(), 4212)

    def test_diel_cycle(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        aggs = diel_cycle(series=s, mincounts=1, mean=True, std=True, median=True, each_month=True)
        months = set(aggs.index.get_level_values(0).tolist())
        self.assertEqual(months, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
        self.assertEqual(aggs.loc[1].sum().sum(), 1235.2002345850228)
        self.assertEqual(aggs.loc[6].sum().sum(), 4928.0285111555195)
        self.assertEqual(aggs.loc[12].sum().sum(), 1043.884056104728)
