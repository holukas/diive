import unittest

import diive.configs.exampledata as ed
from diive.core.times.resampling import diel_cycle


class TestResampling(unittest.TestCase):

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
