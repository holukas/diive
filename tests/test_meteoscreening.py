import unittest

import numpy as np
import pandas as pd

from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

FIELD = 'SWC_FF1_0.05_1'
TAGS = {'units': '%', 'varname': FIELD, 'site': 'CH-LAE', 'hpos': 'FF1', 'vpos': '0.05',
        'repl': '1', 'freq': '10min', 'data_version': 'raw', 'gain': '100.0', 'offset': '0.0'}


def _detailed(index) -> pd.DataFrame:
    """Build a data_detailed frame (one variable plus its database tags)."""
    index.name = 'TIMESTAMP_END'
    df = pd.DataFrame(index=index)
    df[FIELD] = np.linspace(30.0, 35.0, len(index))
    for tag, value in TAGS.items():
        df[tag] = value
    return df


def _screening(df: pd.DataFrame) -> StepwiseMeteoScreeningDb:
    return StepwiseMeteoScreeningDb(site='ch-lae', data_detailed={FIELD: df}, fields=[FIELD],
                                    site_lat=47.478333, site_lon=8.364389, utc_offset=1)


class TestMixedTimeResolution(unittest.TestCase):
    """Input whose raw resolution changes partway through, e.g. after a logger
    program change. All records share the finest grid, each only at its own END."""

    def _mixed(self):
        # Two days at 10MIN followed by two days at 1MIN.
        coarse = pd.date_range('2020-04-10 15:00', '2020-04-12 15:00', freq='10min')
        fine = pd.date_range('2020-04-12 15:01', '2020-04-14 15:00', freq='1min')
        return _detailed(coarse.union(fine)), coarse, fine

    def test_mixed_resolutions_share_the_finest_grid(self):
        df, coarse, fine = self._mixed()
        out = _screening(df).data_detailed[FIELD]

        # Everything ends up on the finest resolution, on middle timestamps.
        self.assertEqual(out.index.freqstr, 'min')
        self.assertEqual(out.index.name, 'TIMESTAMP_MIDDLE')

        # Each coarse record sits only in the grid slot of its END timestamp,
        # unchanged, and the slots between coarse records are empty.
        early = out.loc[:'2020-04-12 14:59:30']
        records = early[FIELD].dropna()
        self.assertEqual(list(records.index), list(coarse - pd.Timedelta('30s')))
        np.testing.assert_array_equal(records.to_numpy(), df.loc[coarse, FIELD].to_numpy())
        self.assertEqual(len(early), 10 * (len(coarse) - 1) + 1)

        # FREQ_AUTO_SEC marks the records with their resolution, empty slots with NaN.
        freq = early['FREQ_AUTO_SEC'].dropna()
        self.assertTrue((freq == 600).all())
        pd.testing.assert_index_equal(freq.index, records.index)

        # The fine era is complete.
        late = out.loc['2020-04-12 15:00:30':]
        self.assertEqual(late[FIELD].notna().sum(), len(fine))
        self.assertTrue((late['FREQ_AUTO_SEC'] == 60).all())

    def test_tags_ignore_empty_slots(self):
        df, _, _ = self._mixed()
        self.assertEqual(_screening(df).tags[FIELD]['site'], TAGS['site'])

    def test_single_resolution_is_left_alone(self):
        """The common case: one resolution, so no upsampling happens."""
        index = pd.date_range('2020-04-12 15:01', '2020-04-14 15:00', freq='1min')
        df = _detailed(index)
        out = _screening(df).data_detailed[FIELD]
        self.assertEqual(out.index.freqstr, 'min')
        self.assertEqual(len(out), len(index))
        self.assertEqual(out[FIELD].isna().sum(), 0)



def _series_detailed(index, values) -> pd.DataFrame:
    """data_detailed frame with given values on a TIMESTAMP_END index."""
    df = _detailed(pd.DatetimeIndex(index))
    df[FIELD] = values
    return df


def _spiky(n_days: int = 3):
    """10MIN data with one spike at a known END timestamp."""
    index = pd.date_range('2020-06-01 00:10', periods=144 * n_days, freq='10min')
    values = 30 + np.random.default_rng(0).normal(0, 0.5, len(index))
    values[100] = 90
    return _series_detailed(index, values), index[100]


class TestResampleMixedResolution(unittest.TestCase):
    """Each record counts once in a sum and by the time it covers in a mean."""

    def _screened(self):
        # 1 mm per 10 min, then 0.1 mm per 1 min: the same rate in both eras.
        coarse = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        fine = pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')
        values = np.r_[np.full(len(coarse), 1.0), np.full(len(fine), 0.1)]
        m = _screening(_series_detailed(coarse.append(fine), values))
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        return m

    def test_sum_counts_each_original_record_once(self):
        m = self._screened()
        m.resample(to_freqstr='30min', agg='sum')
        r = m.resampled_detailed[FIELD][FIELD]
        self.assertAlmostEqual(r.loc['2020-06-02 12:00'], 3.0)  # three 10MIN records
        self.assertAlmostEqual(r.loc['2020-06-04 12:00'], 3.0)  # thirty 1MIN records

    def test_mean_is_unchanged(self):
        m = self._screened()
        m.resample(to_freqstr='30min', agg='mean')
        r = m.resampled_detailed[FIELD][FIELD]
        self.assertAlmostEqual(r.loc['2020-06-02 12:00'], 1.0)
        self.assertAlmostEqual(r.loc['2020-06-04 12:00'], 0.1)

    def _half_hour(self):
        """(12:00, 12:30] holds one 10MIN record (END 12:10) and five 1MIN records (END 12:21-12:25)."""
        index = (pd.date_range('2020-06-01 00:10', '2020-06-02 12:10', freq='10min')
                 .append(pd.date_range('2020-06-02 12:21', '2020-06-02 12:25', freq='1min'))
                 .append(pd.date_range('2020-06-02 12:31', '2020-06-04 00:00', freq='1min')))
        df = _series_detailed(index, np.arange(len(index), dtype=float))
        m = _screening(df)
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        v10 = df.loc['2020-06-02 12:10', FIELD]
        v1 = df.loc['2020-06-02 12:21':'2020-06-02 12:25', FIELD]
        return m, v10, v1

    def test_half_hour_with_both_resolutions(self):
        m, v10, v1 = self._half_hour()
        m.resample(to_freqstr='30min', agg='mean')
        self.assertAlmostEqual(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 12:30'],
                               (10 * v10 + v1.sum()) / 15)
        m.resample(to_freqstr='30min', agg='sum')
        self.assertAlmostEqual(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 12:30'],
                               v10 + v1.sum())

    def test_half_hour_coverage_is_one_half(self):
        m, _, _ = self._half_hour()
        m.resample(to_freqstr='30min', mincounts_perc=.5)
        self.assertFalse(np.isnan(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 12:30']))
        m.resample(to_freqstr='30min', mincounts_perc=.55)
        self.assertTrue(np.isnan(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 12:30']))


class TestMixedResolutionRecords(unittest.TestCase):
    """A coarse record exists in one grid slot only, so tests, removals and
    corrections act on whole records and never fill the empty slots."""

    @staticmethod
    def _index():
        return (pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
                .append(pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')))

    def test_missing_values_flag_real_gaps_only(self):
        # Drop the 10-min record ending 01:00 and the 1-min record ending 06:00 on day 3.
        index = self._index().drop([pd.Timestamp('2020-06-01 01:00'), pd.Timestamp('2020-06-03 06:00')])
        m = _screening(_series_detailed(index, np.ones(len(index))))
        m.flag_missingvals_test()
        flag = m.data_detailed[FIELD].filter(like='MISSING').iloc[:, 0]
        missing_end = flag[flag == 2].index + pd.Timedelta('30s')
        expected = (pd.date_range('2020-06-01 00:51', '2020-06-01 01:00', freq='1min')
                    .append(pd.DatetimeIndex(['2020-06-03 06:00'])))
        self.assertEqual(list(missing_end), list(expected))
        # Empty slots inside a present 10-min record are not flagged at all.
        self.assertTrue(flag.loc['2020-06-01 00:10':'2020-06-01 00:18'].isna().all())

    def test_resample_accepts_old_minute_alias(self):
        m = _screening(_series_detailed(self._index(), np.ones(len(self._index()))))
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        m.resample(to_freqstr='30T')
        self.assertEqual(m.resampled_detailed[FIELD].index.freqstr, '30min')

    def _mixed(self, values=None):
        index = self._index()
        if values is None:
            hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
            values = 30 + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
            values += np.random.default_rng(1).normal(0, .1, len(index))
        df = _series_detailed(index, values)
        return _screening(df), df

    def test_manual_removal_removes_whole_coarse_record(self):
        """L177: removing END 20:00 left nine back-filled copies of it in the aggregate."""
        m, df = self._mixed()
        m.start_outlier_detection()
        m.flag_manualremoval_test(remove_dates=['2020-06-01 20:00'])
        m.addflag()
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(int(cleaned.loc['2020-06-01 19:50:30':'2020-06-01 19:59:30'].notna().sum()), 0)
        m.resample(to_freqstr='30min', agg='mean')
        kept = df.loc[['2020-06-01 19:40', '2020-06-01 19:50'], FIELD].mean()
        self.assertAlmostEqual(m.resampled_detailed[FIELD][FIELD].loc['2020-06-01 20:00'], kept)

    def test_hampel_differencing_keeps_clean_coarse_records(self):
        """L178: back-filled copies gave runs of zero differences, and valid
        10MIN records were rejected."""
        m, _ = self._mixed()
        m.start_outlier_detection()
        m.flag_outliers_hampel_test(window_length=1440, n_sigma=5.5, use_differencing=True)
        flag = m.outlier_detection[FIELD].last_flag
        self.assertEqual(int((flag.loc[:'2020-06-03 00:00'] == 2).sum()), 0)

    def test_corrections_do_not_fill_empty_slots(self):
        index = self._index()
        hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
        radiation = np.clip(800 * np.sin(np.pi * (hour - 5) / 15), 0, None) - 3
        m, _ = self._mixed(values=radiation)
        empty = m.series_hires_orig[FIELD].isna()
        self.assertGreater(int(empty.sum()), 0)

        # The zero offset correction sets every nighttime slot to 0.
        m.correction_remove_nighttime_zero_offset(showplot=False)
        self.assertEqual(int(m.series_hires_cleaned[FIELD][empty].notna().sum()), 0)
        self.assertGreater(int((m.series_hires_cleaned[FIELD] == 0).sum()), 0)

        m.start_outlier_detection()
        m.correction_setto_value(dates=[['2020-06-01', '2020-06-02']], value=5)
        self.assertEqual(int(m.outlier_detection[FIELD].series_hires_cleaned[empty].notna().sum()), 0)
        # Records END 2020-06-01 00:10 to 2020-06-02 23:50
        self.assertEqual(int((m.series_hires_cleaned[FIELD] == 5).sum()), 2 * 144 - 1)

        m.finalize_outlier_detection()
        self.assertEqual(int(m.series_hires_cleaned[FIELD][empty].notna().sum()), 0)


class TestConstruction(unittest.TestCase):

    def test_fields_as_string(self):
        df, _ = _spiky()
        m = StepwiseMeteoScreeningDb(site='ch-lae', data_detailed={FIELD: df}, fields=FIELD,
                                     site_lat=47.478333, site_lon=8.364389, utc_offset=1)
        self.assertEqual(m.fields, [FIELD])

    def test_caller_frames_untouched(self):
        df, _ = _spiky()
        before = df.copy()
        _screening(df)
        pd.testing.assert_frame_equal(df, before)

    def test_tags_skip_missing_values(self):
        """Tables with different tag sets leave some tags empty after merging."""
        df, _ = _spiky()
        df['gain'] = df['gain'].astype(object)
        df.iloc[:10, df.columns.get_loc('gain')] = np.nan
        self.assertEqual(_screening(df).tags[FIELD]['gain'], '100.0')


class TestOutlierWorkflow(unittest.TestCase):

    def test_finalize_removes_rejected_and_rerun_replaces_qcf(self):
        df, spike_end = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        m.flag_outliers_abslim_test(minval=0, maxval=100)
        m.addflag()
        m.finalize_outlier_detection()
        self.assertEqual(m.series_hires_cleaned[FIELD].isna().sum(), 0)

        m.flag_outliers_abslim_test(minval=0, maxval=50)
        m.addflag()
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        spike_middle = spike_end - pd.Timedelta('5min')
        self.assertTrue(np.isnan(cleaned.loc[spike_middle]))
        self.assertEqual(cleaned.isna().sum(), 1)
        dd = m.data_detailed[FIELD]
        self.assertFalse(dd.columns.duplicated().any())
        self.assertEqual(int((dd[f'FLAG_METSCR_{FIELD}_QCF'] == 2).sum()), 1)

    def test_addflag_twice_adds_flag_once(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        m.flag_outliers_abslim_test(minval=0, maxval=50)
        m.addflag()
        m.addflag()
        self.assertEqual(len(m.outlier_detection[FIELD].flags.columns), 1)

    def test_accept_qcf_below_rejects_soft_flags(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        soft = pd.Series(0.0, index=sod.flags.index, name=f'FLAG_{FIELD}_SOFT_TEST')
        soft.iloc[:50] = 1
        sod._flags[soft.name] = soft
        m.finalize_outlier_detection()
        self.assertEqual(m.series_hires_cleaned[FIELD].iloc[:50].isna().sum(), 0)
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        self.assertEqual(m.series_hires_cleaned[FIELD].iloc[:50].isna().sum(), 50)

    def test_manual_removal_takes_end_timestamps(self):
        df, spike_end = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        spike_middle = spike_end - pd.Timedelta('5min')
        # A single END timestamp removes exactly that record.
        m.flag_manualremoval_test(remove_dates=[str(spike_end)])
        flag = m.outlier_detection[FIELD].last_flag
        self.assertEqual(list(flag[flag == 2].index), [spike_middle])
        # A range of END timestamps removes the records ending in it, no half-period shift.
        start, end = df.index[10], df.index[12]
        m.flag_manualremoval_test(remove_dates=[[str(start), str(end)]])
        flag = m.outlier_detection[FIELD].last_flag
        expected = df.index[10:13] - pd.Timedelta('5min')
        self.assertEqual(list(flag[flag == 2].index), list(expected))
        # A bare date covers all records whose END timestamp falls on that day.
        m.flag_manualremoval_test(remove_dates=['2020-06-02'])
        flag = m.outlier_detection[FIELD].last_flag
        self.assertEqual(int((flag == 2).sum()), 144)
        self.assertEqual(flag[flag == 2].index[0], pd.Timestamp('2020-06-01 23:55'))  # END 00:00

    def test_setto_value_takes_end_timestamps(self):
        df, _ = _spiky()
        m = _screening(df)
        start, end = df.index[10], df.index[12]
        m.correction_setto_value(dates=[[str(start), str(end)]], value=-1)
        series = m.series_hires_cleaned[FIELD]
        expected = df.index[10:13] - pd.Timedelta('5min')
        self.assertEqual(list(series[series == -1].index), list(expected))


class TestCorrectionsOrder(unittest.TestCase):
    """Corrections before outlier detection must survive finalize and be what the
    tests see; the notebook order (tests, finalize, corrections) is unchanged."""

    def _tests_and_finalize(self, m):
        m.flag_outliers_abslim_test(minval=0, maxval=50)
        m.addflag()
        m.finalize_outlier_detection()

    def test_correction_before_start(self):
        df, _ = _spiky()
        m = _screening(df)
        m.correction_setto_max_threshold(threshold=31, showplot=False)
        m.start_outlier_detection()
        self._tests_and_finalize(m)
        # The spike was capped to 31 before the tests ran, so nothing is rejected.
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(cleaned.isna().sum(), 0)
        self.assertLessEqual(cleaned.max(), 31)

    def test_correction_after_start_before_tests(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        m.flag_outliers_abslim_test(minval=0, maxval=50)
        m.addflag()  # removes the spike
        m.correction_setto_min_threshold(threshold=30, showplot=False)
        sod_series = m.outlier_detection[FIELD].series_hires_cleaned
        self.assertEqual(sod_series.isna().sum(), 1)  # committed removal kept
        self.assertGreaterEqual(sod_series.min(), 30)
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(cleaned.isna().sum(), 1)
        self.assertGreaterEqual(cleaned.min(), 30)

    def test_notebook_order_matches_qcf_filtered_series(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        self._tests_and_finalize(m)
        qcf_filtered = m.outlier_detection_qcf[FIELD].filteredseries
        pd.testing.assert_series_equal(m.series_hires_cleaned[FIELD], qcf_filtered)
        m.correction_setto_max_threshold(threshold=31, showplot=False)
        expected = qcf_filtered.clip(upper=31)
        pd.testing.assert_series_equal(m.series_hires_cleaned[FIELD], expected, check_names=False)


class TestTimeSpanWindows(unittest.TestCase):
    """Window arguments given as a time span reach the detectors unchanged and
    flag what the equivalent record count flags (144 records = 1 day of 10MIN)."""

    def _flags(self, window):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        m.flag_outliers_hampel_test(window_length=window, repeat=False)
        hampel = sod.last_flag
        m.flag_outliers_localsd_test(n_sd=4, winsize=window, winsize_nighttime=window,
                                     separate_day_night=True, repeat=False)
        localsd = sod.last_flag
        m.flag_outliers_zscore_rolling_test(thres_zscore=4, winsize=window, repeat=False)
        return hampel, localsd, sod.last_flag

    def test_span_matches_record_count(self):
        for by_span, by_records in zip(self._flags('1D'), self._flags(144)):
            self.assertGreater(int((by_records == 2).sum()), 0)
            pd.testing.assert_series_equal(by_span, by_records)

    def test_span_not_a_multiple_of_the_frequency_raises(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        with self.assertRaisesRegex(ValueError, 'whole multiple'):
            m.flag_outliers_localsd_test(winsize='15min')


if __name__ == '__main__':
    unittest.main()
