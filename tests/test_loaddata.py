import unittest

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
