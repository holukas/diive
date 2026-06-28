import unittest

import numpy as np
import pandas as pd

from diive.core.dfun.frames import keep_records_where


class TestKeepRecordsWhere(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'NEE': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0],
            'TA': [5.0, 15.0, 25.0, np.nan, 12.0, 30.0, 8.0],
        })

    def test_set_to_nan_default(self):
        # Default keeps full index, out-of-range -> NaN
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=10, upper=20)
        self.assertEqual(len(out), len(self.df))
        # Only TA=15 (idx 1) and TA=12 (idx 4) fall within [10, 20]
        self.assertEqual(out.notna().sum(), 2)
        self.assertEqual(out.iloc[1], 2.0)
        self.assertEqual(out.iloc[4], 5.0)
        self.assertTrue(np.isnan(out.iloc[0]))

    def test_drop_non_matching(self):
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=10, upper=20, set_to_nan=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.tolist(), [2.0, 5.0])

    def test_open_upper_bound(self):
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=10, set_to_nan=False)
        # TA in {15, 25, 12, 30} >= 10 -> NEE {2, 3, 5, nan}
        self.assertEqual(len(out), 4)

    def test_nan_condition_never_kept(self):
        # idx 3 has TA=NaN; even with a wide range it must not be kept
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=-100, upper=100)
        self.assertTrue(np.isnan(out.iloc[3]))

    def test_invert_removes_in_range(self):
        # invert keeps records OUTSIDE the range (removes the in-range ones).
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=10, upper=20, invert=True, set_to_nan=False)
        # In-range TA = {15, 12} (idx 1, 4) removed; all others kept (incl. the
        # NaN-condition idx 3 and idx 5 whose TA=30 is out of range).
        self.assertEqual(list(out.index), [0, 2, 3, 5, 6])

    def test_invert_keeps_nan_condition(self):
        # A missing condition can't be "in the removed range", so it stays.
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=-100, upper=100, invert=True)
        self.assertEqual(out.iloc[3], 4.0)  # idx 3: TA is NaN -> kept

    def test_inclusive_neither(self):
        out = keep_records_where(self.df, target='NEE', condition_var='TA',
                                 lower=12, upper=15, inclusive='neither',
                                 set_to_nan=False)
        # Boundaries 12 and 15 excluded -> nothing in between
        self.assertEqual(len(out), 0)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            keep_records_where(self.df, target='NEE', condition_var='NOPE', lower=0)

    def test_no_limits_raises(self):
        with self.assertRaises(ValueError):
            keep_records_where(self.df, target='NEE', condition_var='TA')

    def test_input_not_mutated(self):
        before = self.df.copy()
        keep_records_where(self.df, target='NEE', condition_var='TA', lower=10, upper=20)
        pd.testing.assert_frame_equal(self.df, before)


if __name__ == '__main__':
    unittest.main()
