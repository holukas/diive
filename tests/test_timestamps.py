import unittest

import diive.configs.exampledata as ed
from diive.core.times.times import DetectFrequency


class TestTime(unittest.TestCase):

    def test_detect_freq(self):
        df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')

        df = ed.load_exampledata_parquet()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')


if __name__ == '__main__':
    unittest.main()
