import unittest

import numpy as np
import pandas as pd

from diive.core.dfun.frames import keep_records_where, transform_yearmonth_matrix_to_longform


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

    def test_open_bound_stays_open_for_all_inclusive(self):
        # An unset bound is open regardless of 'inclusive'. Substituting the
        # observed min/max instead of infinity made an exclusive setting drop the
        # extreme record of the open side.
        df = pd.DataFrame({'NEE': [1.0, 2.0, 3.0, 4.0, 5.0],
                           'TA': [10.0, 20.0, 30.0, 40.0, 50.0]})
        # Open below: TA=10 must survive in every case.
        expected_open_lower = {'both': [1.0, 2.0, 3.0, 4.0],
                               'neither': [1.0, 2.0, 3.0],
                               'left': [1.0, 2.0, 3.0],
                               'right': [1.0, 2.0, 3.0, 4.0]}
        for inclusive, expected in expected_open_lower.items():
            with self.subTest(bound='open lower', inclusive=inclusive):
                out = keep_records_where(df, target='NEE', condition_var='TA',
                                         upper=40, inclusive=inclusive, set_to_nan=False)
                self.assertEqual(out.tolist(), expected)
        # Open above: TA=50 must survive in every case.
        expected_open_upper = {'both': [2.0, 3.0, 4.0, 5.0],
                               'neither': [3.0, 4.0, 5.0],
                               'left': [2.0, 3.0, 4.0, 5.0],
                               'right': [3.0, 4.0, 5.0]}
        for inclusive, expected in expected_open_upper.items():
            with self.subTest(bound='open upper', inclusive=inclusive):
                out = keep_records_where(df, target='NEE', condition_var='TA',
                                         lower=20, inclusive=inclusive, set_to_nan=False)
                self.assertEqual(out.tolist(), expected)

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


class TestNoCallerMutation(unittest.TestCase):
    """Helpers that return a dataframe must not also modify the one passed in.

    Both of these added their column to the caller's dataframe and returned
    that same object, so they looked pure but were not.
    """

    @staticmethod
    def _df():
        idx = pd.date_range('2024-01-01', periods=10, freq='30min', name='TIMESTAMP_MIDDLE')
        return pd.DataFrame({'TA': np.arange(10.0), 'SW_IN': np.arange(10.0)}, index=idx)

    def test_add_continuous_record_number_leaves_caller_alone(self):
        from diive.core.dfun.frames import add_continuous_record_number
        df = self._df()
        before = list(df.columns)
        out = add_continuous_record_number(df=df, verbose=0)
        self.assertEqual(list(df.columns), before)
        self.assertIsNot(out, df)
        self.assertIn('.RECORDNUMBER', out.columns)
        self.assertEqual(out['.RECORDNUMBER'].iloc[0], 1)
        self.assertEqual(out['.RECORDNUMBER'].iloc[-1], len(df))

    def test_lagged_variants_leaves_caller_alone(self):
        from diive.variables import lagged_variants
        df = self._df()
        before = list(df.columns)
        out = lagged_variants(df=df, lag=[-1, 1], verbose=0)
        self.assertEqual(list(df.columns), before)
        self.assertIsNot(out, df)
        self.assertIn('.TA-1', out.columns)


class TestYearMonthMatrixToLongform(unittest.TestCase):
    """Any year-index/month-column matrix must convert, not only the YEAR/MONTH one."""

    @staticmethod
    def _matrix():
        # Documented input: years as index, months as columns.
        return pd.DataFrame(np.arange(24.0).reshape(2, 12),
                            index=[1997, 1998], columns=range(1, 13))

    def _assert_matches_matrix(self, series, matrixdf):
        first = f'{matrixdf.index[0]}-{matrixdf.columns[0]:02d}-01'
        last = f'{matrixdf.index[-1]}-{matrixdf.columns[-1]:02d}-01'
        self.assertEqual(len(series), matrixdf.size)
        self.assertEqual(series.index.freqstr, 'MS')
        self.assertEqual(series.loc[first], matrixdf.iloc[0, 0])
        self.assertEqual(series.loc[last], matrixdf.iloc[-1, -1])

    def test_unnamed_axes(self):
        matrixdf = self._matrix()
        out = transform_yearmonth_matrix_to_longform(matrixdf=matrixdf, z_var_name='TA')
        self.assertEqual(out.name, 'TA')
        self._assert_matches_matrix(out, matrixdf)

    def test_other_axis_names(self):
        matrixdf = self._matrix().rename_axis(index='year', columns='month')
        out = transform_yearmonth_matrix_to_longform(matrixdf=matrixdf)
        self.assertEqual(out.name, 'VALUE')
        self._assert_matches_matrix(out, matrixdf)

    def test_axis_names_not_changed_for_caller(self):
        matrixdf = self._matrix().rename_axis(index='year', columns='month')
        transform_yearmonth_matrix_to_longform(matrixdf=matrixdf)
        self.assertEqual(matrixdf.index.name, 'year')
        self.assertEqual(matrixdf.columns.name, 'month')

    def test_roundtrip_from_resample_to_monthly_agg_matrix(self):
        from diive.core.times.resampling import resample_to_monthly_agg_matrix
        idx = pd.date_range('2020-01-01', '2021-12-31', freq='D', name='TIMESTAMP_MIDDLE')
        series = pd.Series(np.arange(len(idx), dtype=float), index=idx, name='TA')
        matrixdf = resample_to_monthly_agg_matrix(series=series, agg='mean')
        out = transform_yearmonth_matrix_to_longform(matrixdf=matrixdf, z_var_name='TA')
        self._assert_matches_matrix(out, matrixdf)
        self.assertAlmostEqual(out.sum(), matrixdf.sum().sum(), places=9)
