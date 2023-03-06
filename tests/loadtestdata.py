import unittest

from pandas import DataFrame

from diive.core.io.filereader import ReadFileType
from diive.core.io.files import load_pickle



class TestLoadData(unittest.TestCase):

    def test_readfiletype(self):
        data_df, metadata_df = loadtestdata()
        self.assertEqual(type(data_df), DataFrame)
        return data_df, metadata_df

    def test_load_pickle(self):
        data_df = load_pickle(filepath='../diive/configs/exampledata/exampledata_CH-DAV_FP2022.5_2022_ID20230206154316_30MIN.diive.csv.pickle')
        self.assertEqual(type(data_df), DataFrame)


if __name__ == '__main__':
    unittest.main()
