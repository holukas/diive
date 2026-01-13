import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import DetectFrequency, insert_timestamp
from diive.core.times.times import vectorize_timestamps


class TestTime(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
