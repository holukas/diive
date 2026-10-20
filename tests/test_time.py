import unittest

import numpy as np
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
from diive.core.times.times import remove_after_date
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

    def test_remove_after_date_leaves_input_unchanged(self):
        ix = pd.date_range('2022-08-01', '2023-09-30', freq='D')
        series = pd.Series(range(len(ix)), index=ix, dtype=float, name='x')
        original = series.copy()

        out = remove_after_date(data=series, yearly_end_date='08-11')

        pd.testing.assert_series_equal(series, original)
        self.assertTrue(((out.index.month < 8)
                         | ((out.index.month == 8) & (out.index.day <= 11))).all())
        self.assertEqual(out.index.max(), pd.Timestamp('2023-08-11'))

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

    def test_vectorize_timestamps_season_is_not_nullable(self):
        """.SEASON must not be a nullable extension dtype.

        A single Int64 column turns the whole frame's .to_numpy() into object
        dtype, which every ML fit downstream (e.g. the fallback model, which
        feeds vectorize_timestamps output straight into the regressor) then has
        to convert back to float.
        """
        idx = pd.date_range('2022-01-01 00:30', periods=48 * 400, freq='30min')
        df = pd.DataFrame({'x': 1.0}, index=idx)
        result_df = vectorize_timestamps(df, verbose=0)

        self.assertEqual(result_df['.SEASON'].dtype, 'float64')
        self.assertNotEqual(result_df.to_numpy().dtype, object)
        # Values are unchanged: all four seasons, and matching the month map.
        self.assertEqual(sorted(result_df['.SEASON'].unique()), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result_df.loc['2022-07-01 00:30', '.SEASON'], 2.0)

    def test_detect_freq(self):
        df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')

        df = ed.load_exampledata_parquet()
        f = DetectFrequency(index=df.index, verbose=True)
        freq = f.get()
        self.assertEqual(freq, '30min')

    def test_detect_freq_percent_matching(self):
        """n timestamps have n-1 intervals, and the reported match must say so.

        The denominator used to be the row count, which reported 99% for a
        perfectly regular 100-row index and 60% (instead of 75%) for the
        one-gap index below.
        """
        from diive.core.times.times import timestamp_infer_freq_from_timedelta

        # Perfectly regular: every interval matches.
        idx = pd.date_range('2022-01-01 00:30', periods=100, freq='30min')
        self.assertEqual(timestamp_infer_freq_from_timedelta(idx), ('30min', '100% occurrence'))
        f = DetectFrequency(index=idx)
        self.assertEqual(f.get(), '30min')
        self.assertEqual(f.percent_matching, 100.0)
        self.assertEqual(f.confidence, 1.0)

        # Three of four intervals are 30min, one is 60min -> 75%, not 60%.
        idx = pd.DatetimeIndex(['2022-01-01 00:30', '2022-01-01 01:00', '2022-01-01 01:30',
                                '2022-01-01 02:00', '2022-01-01 03:00'])
        self.assertEqual(timestamp_infer_freq_from_timedelta(idx), ('30min', '75% occurrence'))
        f = DetectFrequency(index=idx)
        self.assertEqual(f.get(), '30min')
        self.assertEqual(f.percent_matching, 75.0)
        self.assertEqual(f.confidence, 0.75)

    def test_detect_freq_percent_matching_not_rounded(self):
        """The match rate must keep its decimals.

        It used to be parsed back out of the '{:.0f}% occurrence' display string,
        so 999 of 1000 matching intervals reported 100.0 and was indistinguishable
        from a perfectly regular record.
        """
        from diive.core.times.times import timestamp_infer_freq_from_timedelta

        base = pd.date_range('2022-01-01 00:30', periods=1000, freq='30min')
        idx = pd.DatetimeIndex(list(base) + [base[-1] + pd.Timedelta('60min')])
        f = DetectFrequency(index=idx)
        self.assertEqual(f.get(), '30min')
        self.assertEqual(f.detection_method, 'timedelta')
        self.assertAlmostEqual(f.percent_matching, 99.9, places=6)
        self.assertAlmostEqual(f.confidence, 0.999, places=6)
        self.assertNotEqual(f.percent_matching, 100.0)

        # The public function keeps returning its rounded display string.
        self.assertEqual(timestamp_infer_freq_from_timedelta(idx), ('30min', '100% occurrence'))

    def test_detect_freq_two_records(self):
        """Two regular timestamps define one interval, which is enough.

        With the row count as denominator this raised RuntimeError, because
        1 of 2 is not more than 50%.
        """
        idx = pd.date_range('2022-01-01 00:30', periods=2, freq='30min')
        f = DetectFrequency(index=idx)
        self.assertEqual(f.get(), '30min')
        self.assertEqual(f.percent_matching, 100.0)
        self.assertEqual(f.detection_method, 'timedelta')

        # Same for other resolutions.
        self.assertEqual(DetectFrequency(index=pd.date_range('2022-01-01', periods=2, freq='1h')).get(), 'h')

    def test_detect_freq_single_record_raises(self):
        """One timestamp has no interval: a clear error, not a bare KeyError."""
        from diive.core.times.times import timestamp_infer_freq_from_timedelta

        idx = pd.date_range('2022-01-01 00:30', periods=1, freq='30min')
        self.assertEqual(timestamp_infer_freq_from_timedelta(idx), (None, '-not-enough-datarows-'))
        with self.assertRaises(RuntimeError):
            DetectFrequency(index=idx)

    def test_sanitizer_reports_percent_matching(self):
        """The public status number must be the interval match, not row-count based."""
        from diive.core.times.times import TimestampSanitizer
        idx = pd.date_range('2022-01-01 00:30', periods=48, freq='30min', name='TIMESTAMP_END')
        s = pd.Series(1.0, index=idx, name='x')
        status = TimestampSanitizer(data=s, validate_naming=False, output_middle_timestamp=False,
                                    verbose=False).get_status()
        self.assertEqual(status['inferred_frequency'], '30min')
        self.assertEqual(status['frequency_percent_matching'], 100.0)

    def test_resampling_to_30MIN(self):
        df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long()
        resampled_ta = resample_series_to_30MIN(series=df['TA_T1_2_1_Avg'])
        # The input starts at 23:59:30 (MIDDLE): that interval is too sparse and kept as NaN
        self.assertEqual(resampled_ta.index[0], pd.Timestamp('2024-04-01 00:00:00'))
        self.assertTrue(pd.isna(resampled_ta.iloc[0]))
        self.assertEqual(resampled_ta.first_valid_index(), pd.Timestamp('2024-04-01 00:30:00'))
        self.assertEqual(resampled_ta.loc['2024-04-09 13:30:00'], 2.643333333333333)
        self.assertEqual(resampled_ta.loc['2024-04-09 14:00:00'], 2.5)
        self.assertEqual(resampled_ta.index.freqstr, '30min')
        self.assertAlmostEqual(resampled_ta.sum(), 7984.021494252875, places=6)
        resampled_swin = resample_series_to_30MIN(series=df['SW_IN_T1_1_1_Avg'])
        # The input starts at 23:59:30 (MIDDLE): that interval is too sparse and kept as NaN
        self.assertEqual(resampled_swin.index[0], pd.Timestamp('2024-04-01 00:00:00'))
        self.assertTrue(pd.isna(resampled_swin.iloc[0]))
        self.assertEqual(resampled_swin.first_valid_index(), pd.Timestamp('2024-04-01 00:30:00'))
        self.assertEqual(resampled_swin.loc['2024-04-09 13:30:00'], 104.64)
        self.assertEqual(resampled_swin.loc['2024-04-09 14:00:00'], 87.08333333333333)
        self.assertEqual(resampled_swin.index.freqstr, '30min')
        self.assertAlmostEqual(resampled_swin.sum(), 134375.59183908044, places=6)

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


class TestSanitizerOffGridRecords(unittest.TestCase):
    """L314: records that are not on the time grid must not vanish silently.

    The grid started at the first timestamp, so one stray first record moved it
    and every other record was lost, with rows_removed reporting 0.
    """

    def setUp(self):
        from diive.core.utils.console import get_verbosity, set_verbosity, VERBOSE_PROGRESS
        self._saved = get_verbosity()
        set_verbosity(VERBOSE_PROGRESS)

    def tearDown(self):
        from diive.core.utils.console import set_verbosity
        set_verbosity(self._saved)

    def _sanitize(self, df):
        from diive.core.times.times import TimestampSanitizer
        from diive.core.utils.console import console
        with console.capture() as cap:
            ts = TimestampSanitizer(data=df, output_middle_timestamp=False, verbose=False)
        return ts.get(), ts.get_status(), cap.get()

    def _assert_on_grid_values_kept(self, df, out, n_off_grid):
        on_grid = df.index.isin(out.index)
        self.assertEqual(int((~on_grid).sum()), n_off_grid)
        pd.testing.assert_series_equal(out['X'].reindex(df.index[on_grid]), df['X'][on_grid],
                                       check_freq=False)

    def test_stray_first_record(self):
        idx = pd.date_range('2022-01-01 00:30', periods=17520, freq='30min', name='TIMESTAMP_END')
        stray = pd.DatetimeIndex([pd.Timestamp('2022-01-01 00:00:01')], name='TIMESTAMP_END')
        df = pd.DataFrame({'X': np.arange(-1.0, 17520.0)}, index=stray.append(idx))
        out, status, msg = self._sanitize(df)
        self.assertEqual(int(out['X'].notna().sum()), 17520)
        self.assertEqual(out.index[0], pd.Timestamp('2022-01-01 00:30'))
        self.assertEqual(out.index.freqstr, '30min')
        self.assertEqual(status['rows_removed_off_grid'], 1)
        self.assertEqual(status['rows_removed'], 1)
        self.assertEqual(status['rows_added_by_regularization'], 0)
        self.assertIn('1 records are not on the 30min time grid', msg)
        self._assert_on_grid_values_kept(df, out, n_off_grid=1)

    def test_jittered_records(self):
        idx = pd.date_range('2022-01-01 00:30', periods=480, freq='30min', name='TIMESTAMP_END')
        jitter = idx.to_series()
        jitter.iloc[100:110] += pd.Timedelta('1s')
        df = pd.DataFrame({'X': np.arange(480.0)}, index=pd.DatetimeIndex(jitter.values, name='TIMESTAMP_END'))
        out, status, msg = self._sanitize(df)
        self.assertEqual(int(out['X'].notna().sum()), 470)
        self.assertEqual(status['rows_removed_off_grid'], 10)
        self.assertEqual(status['rows_removed'], 10)
        # The ten late records leave ten empty grid slots
        self.assertEqual(status['rows_added_by_regularization'], 10)
        self.assertIn('10 records are not on the 30min time grid', msg)
        self._assert_on_grid_values_kept(df, out, n_off_grid=10)

    def test_mixed_10min_and_1min(self):
        i10 = pd.date_range('2022-01-01 00:10', periods=1440, freq='10min')
        i1 = pd.date_range(i10[-1] + pd.Timedelta('1min'), periods=600, freq='1min')
        df = pd.DataFrame({'X': np.arange(2040.0)}, index=i10.append(i1).rename('TIMESTAMP_END'))
        out, status, msg = self._sanitize(df)
        self.assertEqual(status['inferred_frequency'], '10min')
        self.assertEqual(int(out['X'].notna().sum()), 1500)
        self.assertEqual(status['rows_removed_off_grid'], 540)
        self.assertEqual(status['rows_removed'], 540)
        self.assertEqual(status['net_rows'], -540)
        self.assertIn('540 records are not on the 10min time grid', msg)
        self._assert_on_grid_values_kept(df, out, n_off_grid=540)

    def test_regular_grid_with_offset_is_kept(self):
        """A regular grid that does not start on the full hour keeps every record."""
        idx = pd.date_range('2022-01-01 00:20', periods=100, freq='h', name='TIMESTAMP_END')
        df = pd.DataFrame({'X': np.arange(100.0)}, index=idx)
        out, status, msg = self._sanitize(df)
        pd.testing.assert_frame_equal(out, df.asfreq('h'), check_index_type=False)
        self.assertEqual(status['rows_removed'], 0)
        self.assertNotIn('time grid', msg)

    def test_load_parquet_keeps_records_after_stray_first_record(self):
        import tempfile
        from diive.core.io.files import load_parquet, save_parquet
        from diive.core.utils.console import console
        idx = pd.date_range('2022-01-01 00:30', periods=480, freq='30min', name='TIMESTAMP_END')
        stray = pd.DatetimeIndex([pd.Timestamp('2022-01-01 00:00:01')], name='TIMESTAMP_END')
        df = pd.DataFrame({'X': np.arange(-1.0, 480.0)}, index=stray.append(idx))
        with tempfile.TemporaryDirectory() as d:
            with console.capture() as cap:
                path = save_parquet(filename='stray', data=df, outpath=d)
                out = load_parquet(path, output_middle_timestamp=False)
        self.assertEqual(int(out['X'].notna().sum()), 480)
        self.assertIn('1 records are not on the 30min time grid', cap.get())

    def test_tz_aware_across_dst_keeps_all_records(self):
        """L329: with a step that does not divide one hour evenly, the wall-clock
        position after the daylight saving change won the vote. The grid then
        missed every record."""
        for freq in ['30min', 'h', '2h', '3h', '6h', '12h']:
            with self.subTest(freq=freq):
                idx = pd.date_range('2023-01-01', '2023-12-31', freq=freq, tz='Europe/Zurich',
                                    name='TIMESTAMP_END')
                df = pd.DataFrame({'X': np.arange(float(len(idx)))}, index=idx)
                out, status, msg = self._sanitize(df)
                pd.testing.assert_frame_equal(out, df.asfreq(freq), check_index_type=False)
                self.assertEqual(status['rows_removed'], 0)
                self.assertNotIn('time grid', msg)


class TestDetectFreqGroups(unittest.TestCase):
    """Records next to a timestamp gap must keep the resolution of their regular
    neighbour; meteo screening drops every record left without a group."""

    @staticmethod
    def _groups(ix):
        from diive.core.times.times import detect_freq_groups
        return detect_freq_groups(index=pd.DatetimeIndex(ix, name='TIMESTAMP_END'))

    def test_single_missing_record_keeps_neighbours(self):
        ix = pd.date_range('2024-06-01 00:10', periods=144, freq='10min')
        missing = pd.Timestamp('2024-06-01 08:20')
        g = self._groups(ix[ix != missing])
        self.assertEqual(len(g), 143)
        self.assertFalse(g.isna().any())
        self.assertTrue((g == 600).all())

    def test_1min_with_5min_gap_keeps_all_records(self):
        ix = pd.date_range('2024-06-01 00:01', periods=600, freq='1min')
        g = self._groups(ix.delete(range(100, 104)))
        self.assertFalse(g.isna().any())
        self.assertTrue((g == 60).all())

    def test_lone_record_between_two_gaps(self):
        ix = pd.date_range('2024-06-01 00:10', periods=144, freq='10min')
        g = self._groups(ix.delete([50, 52]))
        self.assertTrue((g == 600).all())

    def test_1min_block_then_10min_block(self):
        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:00', freq='1min')
        b = pd.date_range('2024-06-01 10:10', '2024-06-02 10:00', freq='10min')
        g = self._groups(a.append(b))
        self.assertFalse(g.isna().any())
        self.assertTrue((g.loc[:'2024-06-01 10:00'] == 60).all())
        self.assertTrue((g.loc['2024-06-01 10:10':] == 600).all())
        # TIMESTAMP_END: the record at 10:00 covers 09:59-10:00
        self.assertEqual(g.loc['2024-06-01 10:00'], 60)

    def test_10min_block_then_1min_block(self):
        a = pd.date_range('2024-06-01 00:10', '2024-06-01 10:00', freq='10min')
        b = pd.date_range('2024-06-01 10:01', '2024-06-02 10:00', freq='1min')
        g = self._groups(a.append(b))
        self.assertFalse(g.isna().any())
        # TIMESTAMP_END: the record at 10:00 covers 09:50-10:00
        self.assertTrue((g.loc[:'2024-06-01 10:00'] == 600).all())
        self.assertTrue((g.loc['2024-06-01 10:01':] == 60).all())

    def test_transitional_record_stays_unassigned(self):
        a = pd.date_range('2024-06-01 00:10', '2024-06-01 10:00', freq='10min')
        b = pd.date_range('2024-06-01 10:01', '2024-06-02 10:00', freq='1min')
        odd = pd.DatetimeIndex(['2024-06-01 10:00:07'])
        g = self._groups(a.append(odd).append(b))
        self.assertEqual(g.isna().sum(), 1)
        self.assertTrue(pd.isna(g.loc['2024-06-01 10:00:07']))
        self.assertEqual(g.loc[pd.Timestamp('2024-06-01 10:00')], 600)
        self.assertEqual(g.loc[pd.Timestamp('2024-06-01 10:01')], 60)

    def test_random_holes_keep_one_resolution(self):
        # L254: holes left short runs at 20MIN, 30MIN, ... and records without a group
        ix = pd.date_range('2026-03-01 00:10', '2026-04-01 00:00', freq='10min')
        rng = np.random.default_rng(1)
        for p in (0.1, 0.3, 0.5):
            with self.subTest(missing=p):
                g = self._groups(ix[rng.random(len(ix)) >= p])
                self.assertTrue((g == 600).all())

    def test_change_of_resolution_with_holes(self):
        # Only the record at the change itself may be ambiguous.
        rng = np.random.default_rng(3)
        a = pd.date_range('2024-06-01 00:10', '2024-06-03 10:00', freq='10min')
        b = pd.date_range('2024-06-03 10:01', '2024-06-05 10:00', freq='1min')
        truth = np.r_[np.full(len(a), 600.), np.full(len(b), 60.)]
        for p in (0.1, 0.2, 0.4):
            with self.subTest(missing=p):
                keep = rng.random(len(truth)) >= p
                g = self._groups(a.append(b)[keep])
                self.assertFalse(g.isna().any())
                self.assertLessEqual(int((g.to_numpy() != truth[keep]).sum()), 1)

    def test_genuine_period_of_twelve_records_stays(self):
        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:00', freq='1min')
        m = pd.date_range('2024-06-01 10:10', '2024-06-01 12:00', freq='10min')  # 12 records
        c = pd.date_range('2024-06-01 12:01', '2024-06-01 20:00', freq='1min')
        g = self._groups(a.append(m).append(c))
        self.assertTrue((g.loc['2024-06-01 10:10':'2024-06-01 12:00'] == 600).all())
        self.assertTrue((g.loc[:'2024-06-01 10:00'] == 60).all())
        self.assertTrue((g.loc['2024-06-01 12:01':] == 60).all())

    def test_short_coarse_run_stays_its_own_period(self):
        # L265: holes in 1MIN data cannot give 6 or 11 records at exactly 10MIN
        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:00', freq='1min')
        for n in (6, 11):
            with self.subTest(records=n):
                m = pd.date_range('2024-06-01 10:10', periods=n, freq='10min')
                c = pd.date_range(m[-1] + pd.Timedelta('1min'), periods=400, freq='1min')
                g = self._groups(a.append(m).append(c))
                self.assertTrue((g.loc[m] == 600).all())
                self.assertTrue((g.drop(m) == 60).all())

    def test_short_run_that_holes_explain_joins_its_neighbours(self):
        # 20MIN spacing for 11 records is ordinary in 10MIN data with many holes
        ix = pd.date_range('2024-06-01 00:10', periods=600, freq='10min')
        keep = np.ones(len(ix), dtype=bool)
        keep[200:222:2] = False
        g = self._groups(ix[keep])
        self.assertTrue((g == 600).all())

    def test_short_fine_run_stays_its_own_period(self):
        a = pd.date_range('2024-06-01 00:10', '2024-06-01 10:00', freq='10min')
        b = pd.date_range('2024-06-01 10:01', periods=5, freq='1min')
        c = pd.date_range('2024-06-01 10:10', '2024-06-01 20:00', freq='10min')
        g = self._groups(a.append(b).append(c))
        self.assertTrue((g.loc[b] == 60).all())
        self.assertTrue((g.drop(b) == 600).all())

    def test_holey_1min_data_give_no_spurious_coarse_period(self):
        # L265: passes 2-3 joined runs across 1MIN steps into 120 s periods
        full = pd.date_range('2024-01-01 00:01', periods=60 * 24 * 60, freq='1min')
        rng = np.random.default_rng(0)
        for p in (0.3, 0.5, 0.7):
            with self.subTest(missing=p):
                g = self._groups(full[rng.random(len(full)) > p])
                self.assertTrue((g == 60).all())

    def test_own_period_wins_against_a_coarse_run(self):
        # L283: the last 2MIN record before a 30MIN block, with a 2MIN record missing
        # two places before it, has a lone 2MIN neighbour and got 1800 s
        fine = pd.date_range('2024-07-01 12:01', '2024-07-01 15:13', freq='2min')
        fine = fine.drop(pd.Timestamp('2024-07-01 15:07'))
        coarse = pd.date_range('2024-07-01 15:43', periods=20, freq='30min')
        g = self._groups(fine.append(coarse))
        self.assertTrue((g.loc[fine] == 120).all())
        self.assertTrue((g.loc[coarse] == 1800).all())

    def test_own_period_wins_against_a_fine_run(self):
        # L283: the last record of an off-phase 3 x 5MIN block between 2MIN periods
        # got 120 s from the 2MIN run after it
        f1 = pd.date_range('2024-07-02 12:00', '2024-07-02 16:36', freq='2min')
        blk = pd.DatetimeIndex(['2024-07-02 16:45', '2024-07-02 16:50', '2024-07-02 16:55'])
        f2 = pd.date_range('2024-07-02 16:57', periods=200, freq='2min')
        g = self._groups(f1.append(blk).append(f2))
        self.assertTrue((g.loc[blk] == 300).all())
        self.assertTrue((g.drop(blk) == 120).all())

    def test_short_block_on_its_grid_stays_its_own_period(self):
        # L286: 2-5 ten-minute records between 1MIN periods got 60 s
        for name in ('TIMESTAMP_END', 'TIMESTAMP_START'):
            for tz in (None, 'Etc/GMT-1'):
                for n in range(1, 6):
                    with self.subTest(name=name, tz=tz, records=n):
                        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:00', freq='1min', tz=tz)
                        m = pd.date_range('2024-06-01 10:10', periods=n, freq='10min', tz=tz)
                        c = pd.date_range(m[-1] + pd.Timedelta('1min'), periods=400, freq='1min')
                        if name == 'TIMESTAMP_START':
                            a, m, c = (x - pd.Timedelta(f) for x, f in ((a, '1min'), (m, '10min'), (c, '1min')))
                        from diive.core.times.times import detect_freq_groups
                        g = detect_freq_groups(index=pd.DatetimeIndex(a.append(m).append(c), name=name))
                        # A single record stays ambiguous and gets the finer resolution
                        self.assertTrue((g.loc[m] == (60 if n == 1 else 600)).all())
                        self.assertTrue((g.drop(m) == 60).all())

    def test_short_block_off_its_grid_needs_more_records(self):
        # Off the 10MIN grid, holes in 1MIN data explain up to 5 records
        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:05', freq='1min')
        m = pd.date_range('2024-06-01 10:15', periods=3, freq='10min')
        c = pd.date_range(m[-1] + pd.Timedelta('1min'), periods=400, freq='1min')
        g = self._groups(a.append(m).append(c))
        self.assertTrue((g == 60).all())

    def test_coarse_run_keeps_records_after_a_missing_one(self):
        # L289: 15MIN data, then 12 thirty-minute records with 11:30 missing. The
        # last two had no group before pass 4, the run counted 10 records and
        # joined the 15MIN period.
        q = pd.date_range('2024-01-01 01:45', '2024-01-01 06:00', freq='15min')
        h = pd.date_range('2024-01-01 06:30', '2024-01-01 12:30', freq='30min')
        h = h.drop(pd.Timestamp('2024-01-01 11:30'))
        g = self._groups(q.append(h))
        self.assertTrue((g.loc[q] == 900).all())
        self.assertTrue((g.loc[h] == 1800).all())

    def test_empty_and_single_record_index(self):
        # L291: an empty index raised a shape error
        for ix in ([], ['2024-01-01 00:10']):
            with self.subTest(records=len(ix)):
                g = self._groups(pd.to_datetime(ix))
                self.assertEqual(len(g), len(ix))
                self.assertTrue(g.isna().all())
                self.assertEqual(g.name, 'FREQ_AUTO_SEC')

    def test_own_period_wins_for_timestamp_start(self):
        # Mirror image of the 2MIN/30MIN case: for TIMESTAMP_START the record's own
        # period is the next record
        from diive.core.times.times import detect_freq_groups
        coarse = pd.date_range('2024-07-01 02:00', periods=20, freq='30min')
        fine = pd.date_range(coarse[-1] + pd.Timedelta('30min'), periods=97, freq='2min')
        fine = fine.drop(fine[3])
        g = detect_freq_groups(index=pd.DatetimeIndex(coarse.append(fine), name='TIMESTAMP_START'))
        self.assertTrue((g.loc[fine] == 120).all())
        self.assertTrue((g.loc[coarse] == 1800).all())

    @staticmethod
    def _named(ix, name):
        from diive.core.times.times import detect_freq_groups
        return detect_freq_groups(index=pd.DatetimeIndex(ix, name=name))

    def test_regular_outages_keep_their_resolution(self):
        # L301: alternating spacings (1, 2 / 1, 3 / 1, 4 steps) got no group at all
        for name in ('TIMESTAMP_END', 'TIMESTAMP_START'):
            for res in (60, 600):
                for every, missing in ((3, 1), (4, 2), (5, 3)):
                    with self.subTest(name=name, res=res, every=every, missing=missing):
                        ix = pd.date_range('2024-07-01', periods=3000, freq=f'{res}s')
                        ix = ix[(np.arange(len(ix)) % every) >= missing]
                        self.assertTrue((self._named(ix, name) == res).all())

    def test_partial_regular_outage_keeps_its_records(self):
        # L301: two days of 1MIN data with every 3rd record missing and one hour with
        # 3 of 4 missing kept only the 16 records of that hour
        ix = pd.date_range('2024-07-01 00:01', periods=2 * 1440, freq='1min')
        pos = np.arange(len(ix))
        keep = (pos % 3) != 2
        hour = (pos >= 1440) & (pos < 1500)
        keep[hour] = (pos[hour] % 4) == 0
        g = self._groups(ix[keep])
        self.assertFalse(g.isna().any())
        self.assertGreater(int((g == 60).sum()), len(g) - 20)

    def test_regular_outage_needs_whole_multiples_of_one_spacing(self):
        # Stamps that jitter around 10MIN must not get a common resolution of 1 s
        steps = np.tile([599, 601], 200)
        ix = pd.Timestamp('2024-07-01') + pd.to_timedelta(np.r_[0, np.cumsum(steps)], unit='s')
        g = self._groups(ix)
        self.assertFalse((g == 1).any())

    def test_short_block_next_to_a_gap(self):
        # L302: a reboot gap between 1MIN data and a 10MIN block hid the block. A
        # gap before the block hides the spacing of its first record, so 2 records
        # stay ambiguous there.
        a = pd.date_range('2024-06-01 00:01', '2024-06-01 10:00', freq='1min')
        for n, gap_before, gap_after, block in ((2, False, True, True), (3, True, False, True),
                                                (3, True, True, True), (2, True, False, False)):
            with self.subTest(records=n, gap_before=gap_before, gap_after=gap_after):
                m = pd.date_range('2024-06-01 10:20' if gap_before else '2024-06-01 10:10', periods=n, freq='10min')
                c = pd.date_range(m[-1] + pd.Timedelta('6min' if gap_after else '1min'), periods=400, freq='1min')
                for name in ('TIMESTAMP_END', 'TIMESTAMP_START'):
                    ix, blk = a.append(m).append(c), m
                    if name == 'TIMESTAMP_START':
                        # Mirror image in time: each record keeps its own spacing
                        mirror = pd.Timestamp('2024-06-04')
                        ix, blk = (mirror - (ix - mirror)).sort_values(), mirror - (m - mirror)
                    g = self._named(ix, name)
                    self.assertTrue((g.loc[blk] == (600 if block else 60)).all(), name)
                    self.assertTrue((g.drop(blk) == 60).all(), name)

    def test_short_block_at_the_start_or_end(self):
        # L302: at the edges of the data one side has no finer records. With
        # TIMESTAMP_END the first record has no spacing of its own.
        for n in (2, 3, 4):
            with self.subTest(records=n):
                m = pd.date_range('2024-06-01 00:10', periods=n, freq='10min')
                f = pd.date_range(m[-1] + pd.Timedelta('1min'), periods=400, freq='1min')
                g = self._groups(m.append(f))
                self.assertTrue((g.loc[m] == (600 if n >= 4 else 60)).all())
                self.assertTrue((g.loc[f] == 60).all())
                f = pd.date_range('2024-06-01 00:01', periods=400, freq='1min')
                m = pd.date_range(f[-1] + pd.Timedelta('10min'), periods=n, freq='10min')
                g = self._groups(f.append(m))
                self.assertTrue((g.loc[m] == (600 if n >= 3 else 60)).all())
                self.assertTrue((g.loc[f] == 60).all())

    def test_short_block_with_fewer_than_twelve_records(self):
        # L304: pass 4 returned before the block rule
        ix = pd.date_range('2024-06-01 09:57', periods=4, freq='1min').append(
            pd.DatetimeIndex(['2024-06-01 10:10', '2024-06-01 10:20'])).append(
            pd.date_range('2024-06-01 10:21', periods=4, freq='1min'))
        g = self._groups(ix)
        self.assertEqual(g.tolist(), [60] * 4 + [600] * 2 + [60] * 4)

    def test_stray_stamp_does_not_pull_its_neighbour(self):
        # L304: the record after a stray stamp got the stray's spacing and was dropped
        for name in ('TIMESTAMP_END', 'TIMESTAMP_START'):
            for res, stray in ((60, '2024-07-01 12:30:30'), (600, '2024-07-01 12:05')):
                with self.subTest(name=name, res=res):
                    ix = pd.date_range('2024-07-01', periods=(1440 * 60) // res, freq=f'{res}s')
                    ix = ix.append(pd.DatetimeIndex([stray])).sort_values()
                    g = self._named(ix, name)
                    self.assertEqual(g.loc[stray], res / 2)
                    self.assertTrue((g.drop(pd.Timestamp(stray)) == res).all())

    def _both_orientations(self, ix, expected):
        """Check *expected* (per record, in seconds) for TIMESTAMP_END and for the
        mirror image as TIMESTAMP_START (time reversed, so each record keeps its
        own averaging period)."""
        ix = pd.DatetimeIndex(ix)
        mirrored = pd.Timestamp('2030-01-01') - (ix[::-1] - pd.Timestamp('2020-01-01'))
        for name, stamps, exp in (('TIMESTAMP_END', ix, expected), ('TIMESTAMP_START', mirrored, expected[::-1])):
            with self.subTest(name=name):
                g = self._named(stamps, name).to_numpy()
                np.testing.assert_array_equal(g, exp)

    def test_last_coarse_record_keeps_its_resolution(self):
        # L310: with the record before it missing, the last coarse record before
        # 1MIN data took the 60 s of the record after it
        for coarse, missing in (('30min', ['07:30']), ('10min', ['07:50']), ('30min', ['07:00']),
                                ('30min', ['07:00', '07:30'])):
            with self.subTest(coarse=coarse, missing=missing):
                c = pd.date_range('2024-03-04 00:30', '2024-03-05 08:00', freq=coarse)
                c = c.drop(pd.DatetimeIndex([f'2024-03-05 {m}' for m in missing]))
                f = pd.date_range('2024-03-05 08:01', '2024-03-05 12:00', freq='1min')
                expected = np.r_[np.full(len(c), pd.Timedelta(coarse).total_seconds()), np.full(len(f), 60.)]
                self._both_orientations(c.append(f), expected)

    def test_first_coarse_record_after_holey_fine_data(self):
        # L310 mirror case: the record at 10:00 is one 1MIN step from the record
        # before it, which got no resolution in passes 1-3. It took the 600 s of
        # the record after it once the 10MIN run counted as a period.
        f = pd.date_range('2024-06-03 08:00', '2024-06-03 09:51', freq='1min').append(
            pd.DatetimeIndex(['2024-06-03 09:54', '2024-06-03 09:57', '2024-06-03 09:59', '2024-06-03 10:00']))
        c = pd.date_range('2024-06-03 10:10', periods=30, freq='10min')
        self._both_orientations(f.append(c), np.r_[np.full(len(f), 60.), np.full(len(c), 600.)])

    def test_overlapping_regular_outages_keep_their_resolution(self):
        # L309: every 3rd and every 4th 10MIN record missing (spacings 1, 3, 2, 3,
        # 1, 2 steps), or only records 2 and 5 of every 5 kept (3, 2 steps). Next
        # to 1MIN data all records joined the 1MIN period, alone they had no group.
        pos = np.arange(1440)
        ten = pd.date_range('2024-03-01 00:10', periods=len(pos), freq='10min')
        fine = pd.date_range(ten[-1] + pd.Timedelta('1min'), periods=1440, freq='1min')
        for pattern, keep in (('3rd and 4th missing', ~((pos % 3 == 0) | (pos % 4 == 0))),
                              ('2nd and 5th of 5 kept', (pos % 5 == 1) | (pos % 5 == 4))):
            for with_fine in (False, True):
                with self.subTest(pattern=pattern, with_fine=with_fine):
                    ix = ten[keep].append(fine) if with_fine else ten[keep]
                    expected = np.r_[np.full(int(keep.sum()), 600.), np.full(len(fine) if with_fine else 0, 60.)]
                    self._both_orientations(ix, expected)

    def test_pairs_of_regular_outages_keep_their_records(self):
        # L309: 74 of 151 pairs of outage patterns lost more than 5 % of the
        # records. Only a first or last record may stay without a group.
        import itertools
        pats = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (2, 5), (1, 6), (2, 3)]
        pos = np.arange(3000)
        ix = pd.date_range('2024-01-01 00:10', periods=len(pos), freq='10min')
        for (w1, k1), (w2, k2) in itertools.combinations(pats, 2):
            for j in range(k2):
                keep = ~((pos % k1) < w1) & ~(((pos + j) % k2) < w2)
                if keep.mean() < .2:
                    continue
                with self.subTest(p1=(w1, k1), p2=(w2, k2), offset=j):
                    g = self._groups(ix[keep]).to_numpy()
                    self.assertFalse(np.isnan(g[2:-2]).any())

    def test_regular_outage_does_not_spread_from_a_stray_stamp(self):
        # A stray stamp at 00:24:30 in 1MIN data with 00:21 and 00:24 missing
        # leaves 00:23 with spacings of 60 and 90 s (common divisor 30 s). One
        # such record is no regular outage: 00:22 stays 60 s.
        ix = pd.date_range('2024-07-01', periods=600, freq='1min')
        ix = ix.drop(pd.DatetimeIndex(['2024-07-01 00:21', '2024-07-01 00:24']))
        ix = ix.append(pd.DatetimeIndex(['2024-07-01 00:24:30'])).sort_values()
        g = self._groups(ix)
        self.assertEqual(g.loc[pd.Timestamp('2024-07-01 00:22')], 60)
        self.assertNotEqual(g.loc[pd.Timestamp('2024-07-01 00:23')], 30)


class TestDescribeTimeResolution(unittest.TestCase):
    """Text of the detected time resolution, as shown after a download and in plots."""

    def _describe(self, ix):
        from diive.core.times.times import describe_time_resolution
        return describe_time_resolution(ix)

    def test_regular(self):
        self.assertEqual(self._describe(pd.date_range('2024-01-01', periods=50, freq='30min')), '30min')
        self.assertEqual(self._describe(pd.date_range('2024-01-01', periods=50, freq='1min')), '1min')
        self.assertEqual(self._describe(pd.date_range('2024-01-01', periods=5, freq='D')), '24h')

    def test_missing_records_keep_the_resolution(self):
        ix = pd.date_range('2024-01-01', periods=500, freq='10min')
        self.assertEqual(self._describe(ix.delete([10, 11, 12, 200, 300])), '10min')

    def test_mixed_resolutions_with_counts_in_order(self):
        a = pd.date_range('2024-01-01 00:10', periods=100, freq='10min')
        b = pd.date_range(a[-1] + pd.Timedelta('1min'), periods=300, freq='1min')
        self.assertEqual(self._describe(a.append(b)), '10min (100 records), 1min (300 records)')

    def test_unsorted_index_is_sorted_first(self):
        ix = pd.date_range('2024-01-01', periods=50, freq='30min')
        self.assertEqual(self._describe(ix[::-1]), '30min')

    def test_too_few_records(self):
        self.assertEqual(self._describe(pd.DatetimeIndex([])), 'unknown')
        self.assertEqual(self._describe(pd.DatetimeIndex(['2024-01-01'])), 'unknown')


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
