"""
TEST_BINARY: bit-encoded value extraction
=========================================

Run: pytest tests/test_binary.py -v
"""
import unittest

import numpy as np
import pandas as pd

import diive as dv


class TestEncodedValueSeries(unittest.TestCase):

    def test_extracts_the_bit_subrange_and_keeps_nans(self):
        from diive.io.binary import get_encoded_value_series
        series = pd.Series([250.0, np.nan, 18.0])
        out = get_encoded_value_series(int_series=series, bit_start=4, bit_end=8, gain=6.25)
        np.testing.assert_allclose(out.to_numpy(), [62.5, np.nan, 12.5])

    def test_input_series_is_not_modified(self):
        # Missing values used to be overwritten with 0 in the caller's series.
        from diive.io.binary import get_encoded_value_series
        series = pd.Series([250.0, np.nan, 18.0], name='AGC')
        original = series.copy()
        get_encoded_value_series(int_series=series, bit_start=4, bit_end=8)
        pd.testing.assert_series_equal(series, original)

    def test_exported_from_the_binary_package_and_dv(self):
        import diive.io.binary as binary
        self.assertIn('get_encoded_value_series', binary.__all__)
        self.assertIs(binary.get_encoded_value_series, dv.get_encoded_value_series)


if __name__ == '__main__':
    unittest.main()
