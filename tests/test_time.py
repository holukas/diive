import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.resampling import resample_to_daily_agg
from diive.core.times.times import DetectFrequency, insert_timestamp
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


if __name__ == '__main__':
    unittest.main()
