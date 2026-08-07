import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.resampling import resample_to_daily_agg
from diive.core.times.times import (
    DetectFrequency,
    format_timestamp,
    insert_timestamp,
    validate_timestamp_column_name,
)
from diive.core.times.times import keep_daterange
from diive.core.times.times import vectorize_timestamps


class TestTime(unittest.TestCase):

    def test_resample_to_daily_agg(self):
        df, _ = ed.load_exampledata_DIIVE_CSV_30MIN()
        series = df.iloc[:, 0]
        n_days = len(series.resample('D').mean())

        daily = resample_to_daily_agg(series, agg='mean')
        self.assertEqual(len(daily), n_days)
        self.assertEqual(daily.name, series.name)
        # One value per calendar day, sorted, daily frequency.
        self.assertTrue((daily.index.normalize() == daily.index).all())

        # Aggregation methods are honoured: daily max >= daily mean elementwise.
        daily_max = resample_to_daily_agg(series, agg='max')
        self.assertTrue((daily_max.dropna() >= daily.dropna()).all())

        # Completeness filter keeps at most all days.
        strict = resample_to_daily_agg(series, agg='mean', mincounts_perc=1.0)
        self.assertLessEqual(len(strict), len(daily))

        # Non-datetime index raises.
        with self.assertRaises(TypeError):
            resample_to_daily_agg(series.reset_index(drop=True))

    def test_keep_daterange(self):
        df, _ = ed.load_exampledata_DIIVE_CSV_30MIN()
        n_full = len(df)

        # Closed window (inclusive on both ends).
        start, end = df.index[10], df.index[20]
        sub = keep_daterange(df, start, end)
        self.assertEqual(len(sub), 11)
        self.assertEqual(sub.index.min(), start)
        self.assertEqual(sub.index.max(), end)
        self.assertEqual(len(df), n_full)  # non-destructive

        # Open bounds.
        self.assertEqual(len(keep_daterange(df, start=df.index[5])), n_full - 5)
        self.assertEqual(len(keep_daterange(df, end=df.index[5])), 6)
        self.assertEqual(len(keep_daterange(df)), n_full)  # both None -> full copy

        # Works on a Series too.
        self.assertEqual(len(keep_daterange(df.iloc[:, 0], start, end)), 11)

        # Inverted bounds raise; non-datetime index raises.
        with self.assertRaises(ValueError):
            keep_daterange(df, end, start)
        with self.assertRaises(TypeError):
            keep_daterange(df.reset_index(drop=True), start, end)

    def test_vectorize_timestamps(self):
        df, _ = ed.load_exampledata_DIIVE_CSV_30MIN()
        result_df = vectorize_timestamps(df)
        self.assertIn('.YEAR', result_df.columns)
        self.assertIn('.SEASON_SIN', result_df.columns)
        self.assertIn('.MONTH_SIN', result_df.columns)
        self.assertIn('.WEEK_SIN', result_df.columns)
        self.assertIn('.DOY_SIN', result_df.columns)
        self.assertIn('.HOUR_SIN', result_df.columns)

        result_df = vectorize_timestamps(df, year=False, season=False, month=False, week=False, doy=False, hour=False)
        self.assertEqual(len(result_df.columns), len(df.columns))

        result_df = vectorize_timestamps(df, year=True, season=False, month=False, week=False, doy=False, hour=False)
        self.assertIn(".YEAR", result_df.columns)
        self.assertEqual(result_df[".YEAR"].iloc[0], 2022)

        result_df = vectorize_timestamps(df, year=False, season=False, month=True, week=False, doy=False, hour=False)
        self.assertIn(".MONTH", result_df.columns)
        self.assertIn(".MONTH_SIN", result_df.columns)
        self.assertIn(".MONTH_COS", result_df.columns)

        result_df = vectorize_timestamps(df, verbose=0)
        self.assertGreater(len(result_df.columns), len(df.columns))

        df_without_datetime_index = df.reset_index(drop=True)
        with self.assertRaises(AttributeError):
            vectorize_timestamps(df_without_datetime_index)

        result_df = vectorize_timestamps(df, year=False, season=False, month=False, week=False, doy=False, hour=True)
        self.assertIn(".HOUR", result_df.columns)
        self.assertIn(".HOUR_SIN", result_df.columns)
        self.assertIn(".HOUR_COS", result_df.columns)

    def test_detect_freq(self):
        df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')

        df = ed.load_exampledata_parquet()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')

    def test_resampling_to_30MIN(self):
        df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long()
        resampled_ta = resample_series_to_30MIN(series=df['TA_T1_2_1_Avg'])
        self.assertEqual(resampled_ta.index[0], pd.Timestamp('2024-04-01 00:30:00'))
        self.assertEqual(resampled_ta.loc['2024-04-09 13:30:00'], 2.643333333333333)
        self.assertEqual(resampled_ta.loc['2024-04-09 14:00:00'], 2.5)
        self.assertEqual(resampled_ta.index.freqstr, '30min')
        self.assertEqual(resampled_ta.sum(), 7984.021494252875)
        resampled_swin = resample_series_to_30MIN(series=df['SW_IN_T1_1_1_Avg'])
        self.assertEqual(resampled_swin.index[0], pd.Timestamp('2024-04-01 00:30:00'))
        self.assertEqual(resampled_swin.loc['2024-04-09 13:30:00'], 104.64)
        self.assertEqual(resampled_swin.loc['2024-04-09 14:00:00'], 87.08333333333333)
        self.assertEqual(resampled_swin.index.freqstr, '30min')
        self.assertEqual(resampled_swin.sum(), 134375.59183908044)

    def test_insert_timestamp(self):
        df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long()
        df = insert_timestamp(data=df, convention='end')
        df = insert_timestamp(data=df, convention='start')
        checkdata = df.loc['2024-04-05 19:37:30'].copy()
        self.assertEqual(checkdata['TIMESTAMP_START'], pd.Timestamp('2024-04-05 19:37:00'))
        self.assertEqual(checkdata['TIMESTAMP_END'], pd.Timestamp('2024-04-05 19:38:00'))
        self.assertEqual(checkdata.name, pd.Timestamp('2024-04-05 19:37:30'))

    def test_format_timestamp(self):
        df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long()
        # Index is untouched; the result is a new aligned Series.
        end = format_timestamp(df, convention='end')
        start = format_timestamp(df, convention='start')
        self.assertEqual(df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(end.name, 'TIMESTAMP_END')
        self.assertTrue(end.index.equals(df.index))
        self.assertEqual(end.loc['2024-04-05 19:37:30'], pd.Timestamp('2024-04-05 19:38:00'))
        self.assertEqual(start.loc['2024-04-05 19:37:30'], pd.Timestamp('2024-04-05 19:37:00'))
        # With a format string the values are strftime-formatted strings.
        formatted = format_timestamp(df, convention='end', fmt='%Y%m%d%H%M')
        self.assertEqual(formatted.loc['2024-04-05 19:37:30'], '202404051938')

    def test_validate_timestamp_column_name(self):
        # A reserved name must match the column's convention.
        with self.assertRaises(ValueError):
            validate_timestamp_column_name('TIMESTAMP_END', 'start')
        with self.assertRaises(ValueError):
            validate_timestamp_column_name('TIMESTAMP_START', 'middle')
        # Matching reserved name and any non-reserved name are fine.
        validate_timestamp_column_name('TIMESTAMP_END', 'end')
        validate_timestamp_column_name('TIMESTAMP_START', 'start')
        validate_timestamp_column_name('my_timestamp', 'start')

    def test_insert_timestamp_as_index(self):
        df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long()
        self.assertEqual(df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(df.index.freqstr, 'min')
        df = insert_timestamp(data=df, convention='end', set_as_index=True)
        self.assertEqual(df.index.name, 'TIMESTAMP_END')
        self.assertEqual(df.index.freqstr, 'min')

    def test_sanitizer_sorts_and_deduplicates(self):
        from diive.core.times.times import TimestampSanitizer
        base = pd.date_range('2022-01-01 00:00', periods=5, freq='30min')
        # Unsorted, with one duplicate timestamp.
        idx = pd.DatetimeIndex([base[2], base[0], base[1], base[1], base[3], base[4]],
                               name='TIMESTAMP_END')
        s = pd.Series(range(len(idx)), index=idx, name='x', dtype=float)
        clean = TimestampSanitizer(data=s, validate_naming=False, output_middle_timestamp=False,
                                   regularize=True, verbose=False).get()
        self.assertTrue(clean.index.is_monotonic_increasing)
        self.assertFalse(clean.index.has_duplicates)
        self.assertEqual(len(clean), 5)  # 6 input rows, 1 duplicate removed

    def test_sanitizer_regularizes_gaps(self):
        from diive.core.times.times import TimestampSanitizer
        full = pd.date_range('2022-01-01 00:00', periods=10, freq='30min', name='TIMESTAMP_END')
        # Drop two interior timestamps to create gaps.
        idx = full.delete([4, 7])
        s = pd.Series(1.0, index=idx, name='x')
        clean = TimestampSanitizer(data=s, validate_naming=False, output_middle_timestamp=False,
                                   regularize=True, nominal_freq='30min', verbose=False).get()
        # Regularization restores the continuous 30-min grid; gaps become NaN rows.
        self.assertEqual(len(clean), 10)
        self.assertEqual(int(clean.isna().sum()), 2)
        self.assertEqual(clean.index.freqstr, '30min')

    def test_sanitizer_irregular_raises(self):
        from diive.core.times.times import TimestampSanitizer
        # Highly irregular timestamps: no frequency can be detected.
        idx = pd.DatetimeIndex(['2022-01-01 00:00', '2022-01-01 00:03', '2022-01-01 01:17',
                                '2022-01-02 09:00', '2022-01-05 23:41'], name='TIMESTAMP_END')
        s = pd.Series(range(len(idx)), index=idx, name='x', dtype=float)
        with self.assertRaises(RuntimeError):
            TimestampSanitizer(data=s, validate_naming=False, output_middle_timestamp=False,
                               regularize=True, verbose=False).get()


class TestStlDecompose(unittest.TestCase):
    """Regression tests for `core/times/decomposition_utils.py::stl_decompose`.

    Two real bugs were fixed here and neither had a test: the wrapper never
    passed `period` through to statsmodels' STL (so the cycle length the caller
    asked for was ignored), and it called `STL.fit(weights=...)`, which
    statsmodels does not accept, so any `weights=` call raised.
    """

    PERIOD = 24
    CYCLES = 20

    @classmethod
    def setUpClass(cls):
        import numpy as np
        n = cls.PERIOD * cls.CYCLES
        idx = pd.date_range('2021-01-01', periods=n, freq='h', name='TIMESTAMP')
        t = np.arange(n)
        # A clean 24-step cycle on a linear trend: the seasonal component the
        # decomposition should recover is known exactly (amplitude 20).
        cls.series = pd.Series(
            10 * np.sin(2 * np.pi * t / cls.PERIOD)
            + 0.02 * t
            + np.random.RandomState(0).randn(n) * 0.3,
            index=idx, name='X')

    @staticmethod
    def _decompose(series, **kwargs):
        import warnings
        from diive.core.times.decomposition_utils import stl_decompose
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return stl_decompose(series, **kwargs)

    @staticmethod
    def _lag_autocorr(values, lag):
        import numpy as np
        return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])

    def test_period_is_actually_used(self):
        """The regression: `seasonal` must reach statsmodels as `period`.

        With the period honoured, the recovered seasonal component repeats
        exactly every PERIOD steps. When it was dropped, the component tracked
        whatever statsmodels defaulted to instead — which this separates
        cleanly (0.9999 vs 0.005 lag-PERIOD autocorrelation).
        """
        result = self._decompose(self.series, seasonal=self.PERIOD,
                                 trend=self.PERIOD * 2 + 1)
        seasonal = result['seasonal'].to_numpy()
        self.assertGreater(self._lag_autocorr(seasonal, self.PERIOD), 0.99)
        # And it recovers the true amplitude of 20 (10 * sin, peak to trough).
        self.assertAlmostEqual(float(seasonal.max() - seasonal.min()), 20.0, delta=1.5)

    def test_a_wrong_period_does_not_recover_the_cycle(self):
        # The control for the test above: asking for the wrong cycle length must
        # give a visibly different answer, or the assertion above proves nothing.
        result = self._decompose(self.series, seasonal=7, trend=15)
        seasonal = result['seasonal'].to_numpy()
        self.assertLess(abs(self._lag_autocorr(seasonal, self.PERIOD)), 0.5)

    def test_weights_are_no_longer_offered(self):
        """statsmodels' STL takes no observation weights, so neither does diive.

        They used to be accepted, normalised and then dropped on the floor, while
        `quality_weighted_decompose` and `SeasonalTrendDecomposition.summary()`
        reported that weighting had happened. `robust=` is the real knob.
        """
        import numpy as np
        with self.assertRaises(TypeError):
            self._decompose(self.series, seasonal=self.PERIOD,
                            trend=self.PERIOD * 2 + 1,
                            weights=np.linspace(0.0, 1.0, len(self.series)))

    def test_the_quality_weighting_wrapper_is_gone(self):
        import diive.core.times.decomposition_utils as utils
        from diive.analysis.seasonaltrend import SeasonalTrendDecomposition
        self.assertFalse(hasattr(utils, 'quality_weighted_decompose'))
        std = SeasonalTrendDecomposition(self.series, seasonal_period=self.PERIOD)
        self.assertNotIn('Quality-weighted', std.summary())

    def test_components_are_additive_and_keep_the_index(self):
        result = self._decompose(self.series, seasonal=self.PERIOD,
                                 trend=self.PERIOD * 2 + 1)
        for key in ('seasonal', 'trend', 'residual'):
            with self.subTest(component=key):
                self.assertTrue(result[key].index.equals(self.series.index))
        # STL is additive: the three components must sum back to the input.
        # (The function swaps in an integer index internally, then restores the
        # original — this catches that restoration going wrong.)
        recomposed = result['seasonal'] + result['trend'] + result['residual']
        pd.testing.assert_series_equal(recomposed, self.series, check_names=False,
                                       atol=1e-9)

    def test_trend_window_is_normalised(self):
        # statsmodels requires an odd trend window strictly greater than the
        # period; the wrapper fixes up both rather than passing them through to
        # a raise.
        for label, trend in (('even', self.PERIOD * 2), ('below period', 5)):
            with self.subTest(trend=label):
                result = self._decompose(self.series, seasonal=self.PERIOD, trend=trend)
                self.assertEqual(int(result['trend'].isna().sum()), 0)

    def test_invalid_arguments_raise(self):
        for label, kwargs in (('seasonal < 2', dict(seasonal=1)),
                              ('trend < 3', dict(seasonal=24, trend=2))):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    self._decompose(self.series, **kwargs)

    def test_short_series_warns(self):
        import warnings
        from diive.core.times.decomposition_utils import stl_decompose
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            stl_decompose(self.series.head(30), seasonal=self.PERIOD,
                          trend=self.PERIOD * 2 + 1)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught),
                        'a series shorter than 2 * seasonal should warn')


if __name__ == '__main__':
    unittest.main()


class TestStlSurvivesGaps(unittest.TestCase):
    """A gap must not empty the whole decomposition.

    statsmodels' STL has no NaN handling: it propagates rather than raising, so
    one missing value used to give three all-NaN components, `seasonality_strength
    = 0.0` and a `summary()` full of `nan +/- nan` - while four docstrings promised
    gap tolerance. Gaps are the normal state of EC data.
    """

    PERIOD = 24
    CYCLES = 20

    @classmethod
    def setUpClass(cls):
        import numpy as np
        n = cls.PERIOD * cls.CYCLES
        idx = pd.date_range('2021-01-01', periods=n, freq='h', name='TIMESTAMP')
        t = np.arange(n)
        cls.series = pd.Series(10 * np.sin(2 * np.pi * t / cls.PERIOD) + 0.02 * t,
                               index=idx, name='X')

    def _decompose(self, series):
        import warnings
        from diive.core.times.decomposition_utils import stl_decompose
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return stl_decompose(series, seasonal=self.PERIOD, trend=self.PERIOD * 2 + 1)

    def test_a_single_gap_costs_a_single_record(self):
        import numpy as np
        gappy = self.series.copy()
        gappy.iloc[100] = np.nan
        result = self._decompose(gappy)
        self.assertEqual(result['n_interpolated'], 1)
        for key in ('seasonal', 'trend', 'residual'):
            with self.subTest(component=key):
                self.assertEqual(int(result[key].notna().sum()), len(gappy) - 1)
                # The interpolated value is for the fit only - it is not returned.
                self.assertTrue(pd.isna(result[key].iloc[100]))

    def test_leading_and_trailing_gaps_are_covered_too(self):
        # Plain interpolation leaves the edges untouched, and one NaN reaching
        # statsmodels is enough to poison every component.
        import numpy as np
        gappy = self.series.copy()
        gappy.iloc[:3] = np.nan
        gappy.iloc[-2:] = np.nan
        result = self._decompose(gappy)
        self.assertEqual(result['n_interpolated'], 5)
        self.assertEqual(int(result['trend'].notna().sum()), len(gappy) - 5)

    def test_the_components_still_reconstruct_the_measured_records(self):
        import numpy as np
        gappy = self.series.copy()
        gappy.iloc[50:60] = np.nan
        result = self._decompose(gappy)
        recomposed = result['seasonal'] + result['trend'] + result['residual']
        measured = gappy.notna()
        pd.testing.assert_series_equal(recomposed[measured], gappy[measured],
                                       check_names=False, atol=1e-9)

    def test_an_all_nan_series_says_so(self):
        import numpy as np
        allnan = pd.Series(np.nan, index=self.series.index)
        with self.assertRaises(ValueError):
            self._decompose(allnan)


class TestSeasonalityDetectionDoesNotInvent(unittest.TestCase):
    """A failed detection must not look like a successful one."""

    def test_no_candidate_period_raises_instead_of_returning_365(self):
        # It used to return primary_period=365, secondary [7, 30], strength 0.0 -
        # a plausible-looking result - and the caller then decomposed a 5-point
        # series at period 365.
        from diive.core.times.decomposition_utils import detect_seasonality
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                detect_seasonality(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_a_real_cycle_is_still_detected(self):
        import numpy as np
        from diive.core.times.decomposition_utils import detect_seasonality
        t = np.arange(1000)
        res = detect_seasonality(pd.Series(np.sin(2 * np.pi * t / 50)))
        self.assertAlmostEqual(res['primary_period'], 50, delta=1)
