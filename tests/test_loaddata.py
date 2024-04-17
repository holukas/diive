import unittest

import pandas as pd
from pandas import DataFrame

import diive.configs.exampledata as ed


class TestLoadFiletypes(unittest.TestCase):

    def test_load_exampledata_DIIVE_CSV_30MIN(self):
        """Load diive csv file"""
        data_df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 101)
        self.assertEqual(len(data_df), 1488)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 101)

    def test_load_exampledata_eddypro_fluxnet_CSV_30MIN(self):
        """Load EddyPro _fluxnet_ file"""
        data_df, metadata_df = ed.load_exampledata_eddypro_fluxnet_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 488)
        self.assertEqual(len(data_df), 1488)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 488)
        self.assertEqual(data_df.sum().sum(), 307885541284729.3)
        self.assertEqual(data_df.index[0], pd.Timestamp('2021-07-01 00:15:00'))
        self.assertEqual(data_df.index[999], pd.Timestamp('2021-07-21 19:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2021-07-31 23:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30T')

    def test_load_exampledata_eddypro_full_output_CSV_30MIN(self):
        """Load EddyPro _fluxnet_ file"""
        data_df, metadata_df = ed.load_exampledata_eddypro_full_output_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 120)
        self.assertEqual(len(data_df), 468)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 120)
        self.assertEqual(data_df['co2_flux'].sum(), -1781.7114716)
        self.assertEqual(data_df.sum().sum(), 2998405615966.3564)
        self.assertEqual(data_df.index[0], pd.Timestamp('2024-03-29 01:15:00'))
        self.assertEqual(data_df.index[10], pd.Timestamp('2024-03-29 06:15:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2024-04-07 18:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30T')

    def test_exampledata_pickle(self):
        """Load pickled file"""
        data_df = ed.load_exampledata_pickle()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(len(data_df.columns), 101)
        self.assertEqual(len(data_df), 17520)

    def test_exampledata_parquet(self):
        """Load parquet file"""
        data_df = ed.load_exampledata_parquet()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(len(data_df.columns), 49)
        self.assertEqual(len(data_df), 175296)


if __name__ == '__main__':
    unittest.main()
