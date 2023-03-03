import unittest

from pandas import DataFrame

from diive.core.io.filereader import ReadFileType
from diive.core.io.files import load_pickle

def loadtestdata():
    loaddatafile = ReadFileType(filetype='DIIVE_CSV_30MIN',
                                filepath='../diive/configs/exampledata/exampledata_CH-DAV_FP2022.5_2022.07_ID20230206154316_30MIN.diive.csv',
                                data_nrows=None)
    data_df, metadata_df = loaddatafile._readfile()
    return data_df, metadata_df

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
