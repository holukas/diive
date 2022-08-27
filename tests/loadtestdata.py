import unittest

from pandas import DataFrame
import diive

# from diive.core.io.filereader import ReadFileType
# from diive.core.io.files import save_as_pickle, load_pickle

# import diive.core.io.files as files
# from diive.core.io.filereader import ReadFileType


class TestReadFileType(unittest.TestCase):

    def load_testdata(self):
        loaddatafile = ReadFileType(filetype='REDDYPROC_30MIN',
                                    filepath='testdata/testdata_CH-DAV_FP2021.2_2016-2020_ID20220324003457_30MIN_SUBSET.csv',
                                    data_nrows=None)
        data_df, metadata_df = loaddatafile._readfile()
        return data_df, metadata_df

    def test_readfiletype(self):
        data_df, metadata_df = self.load_testdata()
        self.assertEqual(type(data_df), DataFrame)

    def test_pickle(self):
        data_df, metadata_df = self.load_testdata()
        filepath = save_as_pickle(outpath='',
                                  filename='testdata/testdata_CH-DAV_FP2021.2_2016-2020_ID20220324003457_30MIN_SUBSET.csv',
                                  data=data_df)
        load_pickle(filepath=filepath)


if __name__ == '__main__':
    unittest.main()
