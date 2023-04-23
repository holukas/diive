import unittest

import diive.configs.exampledata as ed
from diive.core.times.times import DetectFrequency


class TestTimestamps(unittest.TestCase):

    def test_detect_freq(self):
        data_df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        f = DetectFrequency(index=data_df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30T')  # add assertion here


if __name__ == '__main__':
    unittest.main()
