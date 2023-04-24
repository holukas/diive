import unittest

import diive.configs.exampledata as ed
from diive.core.times.times import DetectFrequency
from diive.pkgs.formats.fluxnet import ConvertEddyProFluxnetFileForUpload


class TestFormats(unittest.TestCase):

    def test_xxx(self):
        pass
        # data_df, metadata_df = ed.load_exampledata_eddypro_fluxnet_CSV_30MIN()
        #
        # f = DetectFrequency(index=data_df.index, verbose=True)
        # freq = f.get()
        # self.assertEqual(freq, '30T')  # add assertion here


if __name__ == '__main__':
    unittest.main()
