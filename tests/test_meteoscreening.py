import unittest
from unittest import mock

import numpy as np
import pandas as pd

import diive.core.utils.console as console_module
from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb
from diive.variables.radiation import potrad

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

    def test_missing_values_flag_counts_missing_records(self):
        # Drop the 10-min record ending 01:00 and the 1-min record ending 06:00 on day 3.
        index = self._index().drop([pd.Timestamp('2020-06-01 01:00'), pd.Timestamp('2020-06-03 06:00')])
        m = _screening(_series_detailed(index, np.ones(len(index))))
        m.flag_missingvals_test()
        flag = m.data_detailed[FIELD].filter(like='MISSING').iloc[:, 0]
        # Each missing record is flagged once, at its END slot.
        missing_end = flag[flag == 2].index + pd.Timedelta('30s')
        self.assertEqual(list(missing_end), list(pd.DatetimeIndex(['2020-06-01 01:00', '2020-06-03 06:00'])))
        # The other slots of the missing 10-min record, and the empty slots inside
        # a present one, are no record at all.
        self.assertTrue(flag.loc['2020-06-01 00:50:30':'2020-06-01 00:58:30'].isna().all())
        self.assertTrue(flag.loc['2020-06-01 00:10':'2020-06-01 00:18'].isna().all())
        self.assertEqual(int((flag == 0).sum()), len(index))

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


def _diurnal(index, seed=1, sd=.1):
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    return 30 + 3 * np.sin(2 * np.pi * (hour - 9) / 24) + np.random.default_rng(seed).normal(0, sd, len(index))


class _Lines:
    """Console mirror that keeps the printed lines."""

    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(' '.join(str(a) for a in args))

    def log(self, *args, **kwargs):
        self.print(*args)


def _warnings(call) -> list:
    """Run *call* and return the warnings diive printed to the console."""
    out = _Lines()
    console = console_module.console
    console.add_mirror(out)
    try:
        call()
    finally:
        console.remove_mirror(out)
    return [line.split('[/yellow] ', 1)[1] for line in out.lines if '[yellow]!' in line]


class TestPerResolutionPeriod(unittest.TestCase):
    """Rolling-window and difference tests run on each resolution period at its
    own resolution, so the 10MIN part of mixed data is screened like pure 10MIN data."""

    @staticmethod
    def _data():
        coarse = pd.date_range('2020-06-01 00:10', periods=6 * 144, freq='10min')
        fine = pd.date_range(coarse[-1] + pd.Timedelta('1min'), periods=2 * 1440, freq='1min')
        index = coarse.append(fine)
        values = _diurnal(index)
        for i, add in ((150, 5), (400, -5), (700, 4), (820, 5)):
            values[i] += add
        mixed = _series_detailed(index, values)
        pure = mixed.loc[coarse].rename_axis('TIMESTAMP_END')
        return mixed, pure, coarse

    @staticmethod
    def _flag_by_end(df, test):
        m = _screening(df)
        m.start_outlier_detection()
        test(m)
        flag = m.outlier_detection[FIELD].last_flag
        return flag.set_axis(flag.index + pd.Timedelta(flag.index.freq) / 2)

    def test_coarse_part_is_flagged_like_pure_coarse_data(self):
        mixed, pure, coarse = self._data()
        tests = {
            'hampel_diff': lambda m: m.flag_outliers_hampel_test(window_length='1D', n_sigma=5.5,
                                                                 use_differencing=True, repeat=True),
            'hampel_records': lambda m: m.flag_outliers_hampel_test(window_length=13, n_sigma=5.5,
                                                                    use_differencing=False,
                                                                    separate_day_night=False),
            'localsd': lambda m: m.flag_outliers_localsd_test(n_sd=4, winsize='1D', separate_day_night=True,
                                                              repeat=False),
            'zscore_rolling': lambda m: m.flag_outliers_zscore_rolling_test(thres_zscore=3, winsize='3h'),
            'increments': lambda m: m.flag_outliers_increments_zcore_test(thres_zscore=5),
        }
        for name, test in tests.items():
            with self.subTest(test=name):
                expected = self._flag_by_end(pure, test)
                self.assertGreater(int((expected == 2).sum()), 0)
                got = self._flag_by_end(mixed, test).reindex(coarse)
                np.testing.assert_array_equal(got.to_numpy(), expected.reindex(coarse).to_numpy())

    def test_empty_slots_stay_untested_and_flag_is_added(self):
        mixed, _, _ = self._data()
        m = _screening(mixed)
        m.start_outlier_detection()
        m.flag_outliers_hampel_test(window_length='1D')
        sod = m.outlier_detection[FIELD]
        flag = sod.last_flag
        self.assertEqual(flag.name, f'FLAG_{FIELD}_OUTLIER_HAMPEL_TEST')
        self.assertTrue(flag[~m._has_record(FIELD)].isna().all())
        self.assertTrue(flag[m._has_record(FIELD)].notna().all())
        m.addflag()
        self.assertEqual(list(sod.flags.columns), [flag.name])
        self.assertEqual(int(sod.series_hires_cleaned.notna().sum()),
                         int(m._has_record(FIELD).sum() - (flag == 2).sum()))

    def test_short_period_is_tested_with_a_warning(self):
        """Three 1MIN records between two 10MIN periods form a period shorter than '1D'."""
        first = pd.date_range('2020-06-01 00:10', periods=3 * 144, freq='10min')
        short = pd.date_range(first[-1] + pd.Timedelta('1min'), periods=3, freq='1min')
        last = pd.date_range(short[-1] + pd.Timedelta('7min'), periods=3 * 144, freq='10min')
        index = first.append(short).append(last)
        m = _screening(_series_detailed(index, _diurnal(index)))
        m.start_outlier_detection()
        self.assertEqual([len(slots) for slots, _, _ in m._resolution_periods(FIELD)], [432, 3, 432])
        # Default arguments (verbose=False): the warning must still be shown.
        messages = _warnings(lambda: m.flag_outliers_hampel_test(window_length='1D'))
        self.assertEqual(len(messages), 1)
        self.assertIn('1 resolution period(s) shorter than the window', messages[0])
        flag = m.outlier_detection[FIELD].last_flag
        self.assertTrue(flag.loc[short - pd.Timedelta('30s')].notna().all())

    def test_period_with_fewer_than_three_records_stays_untested(self):
        """Two 1MIN records between 10MIN periods: flag NaN, as for a single record."""
        first = pd.date_range('2020-06-01 00:10', periods=2 * 144, freq='10min')
        two = pd.date_range(first[-1] + pd.Timedelta('1min'), periods=2, freq='1min')
        last = pd.date_range(first[-1] + pd.Timedelta('10min'), periods=144, freq='10min')
        index = first.append(two).append(last)
        m = _screening(_series_detailed(index, _diurnal(index)))
        m.start_outlier_detection()
        self.assertEqual([len(slots) for slots, _, _ in m._resolution_periods(FIELD)], [288, 2, 144])
        messages = _warnings(lambda: m.flag_outliers_increments_zcore_test(thres_zscore=5))
        self.assertEqual(len(messages), 1)
        self.assertIn('1 resolution period(s) with fewer than 3 records, left untested', messages[0])
        flag = m.outlier_detection[FIELD].last_flag
        self.assertTrue(flag.loc[two - pd.Timedelta('30s')].isna().all())
        self.assertEqual(int(flag.notna().sum()), 288 + 144)

    def test_no_testable_period_gives_an_empty_named_flag(self):
        """If every period has fewer than three records, the flag is named like the
        test's flag and is NaN everywhere, so addflag() adds a proper column."""
        mixed, _, _ = self._data()
        m = _screening(mixed)
        m.start_outlier_detection()
        # Periods this short hardly survive the resolution detection, so cut them here.
        periods = [(slots[:2], middles[:2], res) for slots, middles, res in m._resolution_periods(FIELD)]
        with mock.patch.object(m, '_resolution_periods', return_value=periods):
            messages = _warnings(lambda: m.flag_outliers_hampel_test(window_length='1D'))
        self.assertEqual(len(messages), 1)
        self.assertIn('all 2 resolution periods have fewer than 3 records', messages[0])
        sod = m.outlier_detection[FIELD]
        self.assertEqual(sod.last_flag.name, f'FLAG_{FIELD}_OUTLIER_HAMPEL_TEST')
        self.assertTrue(sod.last_flag.isna().all())
        m.addflag()
        self.assertEqual(list(sod.flags.columns), [f'FLAG_{FIELD}_OUTLIER_HAMPEL_TEST'])

    def test_span_window_not_fitting_a_period_names_the_period(self):
        mixed, _, _ = self._data()
        m = _screening(mixed)
        m.start_outlier_detection()
        for window in ('15min', '5min'):
            with self.subTest(window=window):
                with self.assertRaisesRegex(ValueError, r'resolution period 1 of 2, .*\(END, 10min\): '
                                                        r".*window='\d+min'.*every period's resolution"):
                    m.flag_outliers_zscore_rolling_test(winsize=window)

    def test_single_resolution_runs_the_test_directly(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        with mock.patch.object(sod, 'set_pending_flag') as hook:
            m.flag_outliers_hampel_test(window_length=144)
        hook.assert_not_called()

    def test_pending_flag_must_match_the_index(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        with self.assertRaisesRegex(ValueError, 'same timestamp index'):
            sod.set_pending_flag(pd.Series(0.0, index=sod.flags.index[:-1], name='FLAG_X_TEST'))


class TestMixedResolutionGrid(unittest.TestCase):

    def test_records_a_second_late_raise(self):
        """Three 10MIN records 1 s (or 1 ms) late would need a grid of 1 s (1 ms)."""
        for late, step in (('1s', '1s'), ('1ms', '1ms')):
            with self.subTest(late=late):
                ends = pd.date_range('2020-06-01 00:10', periods=30 * 144, freq='10min').to_series()
                ends.iloc[2000:2003] += pd.Timedelta(late)
                with self.assertRaises(ValueError) as raised:
                    _screening(_series_detailed(pd.DatetimeIndex(ends), np.ones(len(ends))))
                message = str(raised.exception)
                self.assertIn('3 of 4320 records are off the 10min phase', message)
                self.assertIn(str(ends.iloc[2000]), message)
                self.assertIn(f'a grid of {step} with', message)
                self.assertIn('Clean the timestamps first', message)

    def test_phase_shift_within_the_limit_is_kept(self):
        """10MIN records at :00, then at :05: a 5MIN grid (2 slots per record) holds both."""
        first = pd.date_range('2020-06-01 00:10', periods=2 * 144, freq='10min')
        index = first.append(pd.date_range(first[-1] + pd.Timedelta('15min'), periods=2 * 144, freq='10min'))
        m = _screening(_series_detailed(index, np.ones(len(index))))
        dd = m.data_detailed[FIELD]
        self.assertEqual(dd.index.freqstr, '5min')
        self.assertEqual(int(dd[FIELD].notna().sum()), len(index))

    def test_resolution_that_does_not_divide_the_finest(self):
        """10MIN then 15MIN records: a 5MIN grid keeps every record."""
        index = (pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
                 .append(pd.date_range('2020-06-03 00:15', '2020-06-05 00:00', freq='15min')))
        m = _screening(_series_detailed(index, np.ones(len(index))))
        dd = m.data_detailed[FIELD]
        self.assertEqual(dd.index.freqstr, '5min')
        self.assertEqual(int(dd[FIELD].notna().sum()), len(index))
        m.flag_missingvals_test()
        self.assertEqual(int((dd.filter(like='MISSING').iloc[:, 0] == 2).sum()), 0)
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        m.resample(to_freqstr='30min', agg='sum', mincounts_perc=1)
        r = m.resampled_detailed[FIELD][FIELD]
        self.assertAlmostEqual(r.loc['2020-06-02 12:00'], 3.0)  # three 10MIN records
        self.assertAlmostEqual(r.loc['2020-06-04 12:00'], 2.0)  # two 15MIN records

    def test_overlapping_transition_counts_the_overlap_once(self):
        """1MIN records up to END 00:05, then 10MIN records: the record END 00:10
        covers 00:00-00:10 and overlaps five 1MIN records."""
        index = (pd.date_range('2020-06-01 00:01', '2020-06-02 00:05', freq='1min')
                 .append(pd.date_range('2020-06-02 00:10', '2020-06-03 00:00', freq='10min')))
        values = np.where(index <= pd.Timestamp('2020-06-02 00:05'), 0.0, 10.0)
        m = _screening(_series_detailed(index, values))
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        # Covered exactly once, so even a full-coverage minimum keeps the period.
        m.resample(to_freqstr='30min', agg='mean', mincounts_perc=1)
        self.assertAlmostEqual(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 00:30'],
                               (5 * 0 + 5 * 10 + 20 * 10) / 30)
        m.resample(to_freqstr='30min', agg='sum')
        self.assertAlmostEqual(m.resampled_detailed[FIELD][FIELD].loc['2020-06-02 00:30'], 30.0)


class TestTrueMiddle(unittest.TestCase):

    def test_potential_radiation_of_coarse_records(self):
        """The QCF day/night of a 10MIN record uses its own period, as in pure 10MIN data."""
        coarse = pd.date_range('2020-06-01 00:10', periods=3 * 144, freq='10min')
        fine = pd.date_range(coarse[-1] + pd.Timedelta('1min'), periods=1440, freq='1min')
        mixed = _screening(_series_detailed(coarse.append(fine), np.ones(len(coarse) + len(fine))))
        pure = _screening(_series_detailed(coarse, np.ones(len(coarse))))
        got = mixed._potential_radiation(FIELD).loc[coarse - pd.Timedelta('30s')]
        expected = pure._potential_radiation(FIELD).loc[coarse - pd.Timedelta('5min')]
        np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())
        # The 1MIN part is unchanged.
        np.testing.assert_array_equal(
            mixed._potential_radiation(FIELD).loc[fine - pd.Timedelta('30s')].to_numpy(),
            potrad(timestamp_index=mixed.data_detailed[FIELD].index, lat=47.478333, lon=8.364389,
                   utc_offset=1).loc[fine - pd.Timedelta('30s')].to_numpy())


class TestRefinalize(unittest.TestCase):

    def _soft(self, n_soft: int = 50):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        soft = pd.Series(0.0, index=sod.flags.index, name=f'FLAG_{FIELD}_SOFT_TEST')
        soft.iloc[:n_soft] = 1
        sod.set_pending_flag(soft)
        m.addflag()
        return m

    def test_looser_run_brings_records_back(self):
        m = self._soft()
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        self.assertEqual(int(m.series_hires_cleaned[FIELD].iloc[:50].isna().sum()), 50)
        m.finalize_outlier_detection()
        self.assertEqual(int(m.series_hires_cleaned[FIELD].isna().sum()), 0)

    def test_records_come_back_corrected(self):
        m = self._soft()
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        m.correction_setto_max_threshold(threshold=30, showplot=False)
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(int(cleaned.isna().sum()), 0)
        self.assertLessEqual(cleaned.max(), 30)

    def test_tests_after_a_correction_see_the_removed_records(self):
        """The strict run removes the soft-flagged spike; a test run after a
        correction must still test it, so the looser run does not bring it back."""
        m = self._soft(n_soft=150)  # the spike is record 100
        m.flag_manualremoval_test(remove_dates=['2020-06-01 01:00'])  # record 5
        m.addflag()
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        self.assertEqual(int(m.series_hires_cleaned[FIELD].iloc[:150].isna().sum()), 150)
        m.correction_setto_min_threshold(threshold=0, showplot=False)  # changes no value
        sod = m.outlier_detection[FIELD]
        self.assertEqual(sod.series_hires_cleaned.iloc[100], 90)
        self.assertTrue(np.isnan(sod.series_hires_cleaned.iloc[5]))  # added flag keeps it removed
        m.flag_outliers_abslim_test(minval=0, maxval=50)
        self.assertEqual(sod.last_flag.iloc[100], 2)
        m.addflag()
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(list(np.flatnonzero(cleaned.isna())), [5, 100])

    def test_order_of_corrections_and_finalize(self):
        """Correction before start, between tests and after finalize, then a
        looser finalize: every correction holds for all records it brings back."""
        df, _ = _spiky()
        m = _screening(df)
        m.correction_setto_min_threshold(threshold=29, showplot=False)
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        soft = pd.Series(0.0, index=sod.flags.index, name=f'FLAG_{FIELD}_SOFT_TEST')
        soft.iloc[:50] = 1
        sod.set_pending_flag(soft)
        m.addflag()
        m.correction_setto_max_threshold(threshold=31, showplot=False)  # caps the spike
        m.flag_outliers_abslim_test(minval=0, maxval=50)
        m.addflag()
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        m.correction_setto_max_threshold(threshold=30.5, showplot=False)
        sod_series = m.outlier_detection[FIELD].series_hires_cleaned
        self.assertTrue(sod_series.iloc[:50].notna().all())  # removed by the QCF only
        self.assertLessEqual(sod_series.max(), 30.5)
        m.finalize_outlier_detection()
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(int(cleaned.isna().sum()), 0)
        self.assertGreaterEqual(cleaned.min(), 29)
        self.assertLessEqual(cleaned.max(), 30.5)


class TestResampleDaily(unittest.TestCase):

    def test_daily_target(self):
        single = pd.date_range('2020-06-01 00:10', '2020-06-04 00:00', freq='10min')
        mixed = (pd.date_range('2020-06-01 00:10', '2020-06-02 00:00', freq='10min')
                 .append(pd.date_range('2020-06-02 00:01', '2020-06-04 00:00', freq='1min')))
        for index, records_per_day in ((single, [144, 144, 144]), (mixed, [144, 1440, 1440])):
            m = _screening(_series_detailed(index, np.ones(len(index))))
            m.start_outlier_detection()
            m.finalize_outlier_detection()
            m.resample(to_freqstr='1D', agg='sum')
            r = m.resampled_detailed[FIELD][FIELD]
            self.assertEqual(list(r.index), list(pd.date_range('2020-06-02', '2020-06-04', freq='1D')))
            self.assertEqual(r.tolist(), records_per_day)


class TestUnmatchedDates(unittest.TestCase):

    def test_dates_without_record_warn(self):
        index = (pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
                 .append(pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')))
        m = _screening(_series_detailed(index, np.ones(len(index))))
        m.start_outlier_detection()
        # Inside a 10MIN record's period, a date with no data, and one valid record.
        dates = ['2020-06-01 20:05', ['2021-01-01', '2021-01-02'], '2020-06-01 20:10']
        messages = _warnings(lambda: m.flag_manualremoval_test(remove_dates=dates))
        self.assertEqual(len(messages), 2)
        self.assertIn("'2020-06-01 20:05'", messages[0])
        flag = m.outlier_detection[FIELD].last_flag
        self.assertEqual(list(flag[flag == 2].index), [pd.Timestamp('2020-06-01 20:09:30')])
        messages = _warnings(lambda: m.correction_setto_value(dates=['2020-06-01 20:05'], value=5))
        self.assertEqual(len(messages), 1)
        self.assertEqual(int((m.series_hires_cleaned[FIELD] == 5).sum()), 0)


class TestMissingFlagAtFinalize(unittest.TestCase):

    def test_added_for_mixed_resolutions(self):
        index = (pd.date_range('2020-06-01 00:10', '2020-06-02 00:00', freq='10min')
                 .append(pd.date_range('2020-06-02 00:01', '2020-06-03 00:00', freq='1min')))
        m = _screening(_series_detailed(index, np.ones(len(index))))
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        flag = m.data_detailed[FIELD][f'FLAG_{FIELD}_MISSING_TEST']
        self.assertTrue(flag[~m._has_record(FIELD)].isna().all())
        self.assertEqual(int((flag == 0).sum()), len(index))

    def test_not_added_for_one_record_per_slot(self):
        df, _ = _spiky()
        m = _screening(df)
        m.start_outlier_detection()
        m.finalize_outlier_detection()
        self.assertEqual(len(m.data_detailed[FIELD].filter(like='MISSING').columns), 0)


def _segments(ydata) -> int:
    """Line segments drawn: pairs of neighbouring points that are both finite."""
    finite = np.isfinite(np.asarray(ydata, dtype=float))
    return int((finite[1:] & finite[:-1]).sum())


class TestMixedResolutionPlots(unittest.TestCase):
    """Plots show the coarse part of mixed data as readable as the fine part,
    from display copies that leave the data alone."""

    COARSE_END = pd.Timestamp('2020-06-02 23:59:30')  # last 10MIN record, TIMESTAMP_MIDDLE
    MISSING = pd.Timestamp('2020-06-01 12:00')  # END of a missing 10MIN record
    SPIKE = pd.Timestamp('2020-06-02 06:00')  # END of a 10MIN record the z-score test removes

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def _screened(self, mixed: bool = True):
        index = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        if mixed:
            index = index.append(pd.date_range('2020-06-03 00:01', '2020-06-04 00:00', freq='1min'))
        index = index.drop(self.MISSING)
        values = _diurnal(index)
        values[index.get_loc(self.SPIKE)] += 50
        m = _screening(_series_detailed(index, values))
        m.start_outlier_detection()
        m.flag_outliers_zscore_test(thres_zscore=4)
        m.addflag()
        m.flag_missingvals_test()
        m.finalize_outlier_detection()
        m.resample('30min')
        return m

    @staticmethod
    def _figure(call):
        import matplotlib.pyplot as plt
        plt.close('all')
        call()
        return plt.gcf()

    def _coarse_segments(self, line) -> int:
        x = pd.DatetimeIndex(line.get_xdata())
        return _segments(np.asarray(line.get_ydata(), dtype=float)[x <= self.COARSE_END])

    def test_lines_join_coarse_records_and_break_at_gaps(self):
        m = self._screened()
        # 287 records and the missing record's slot give 288 points, 287 segments
        # if the line were unbroken. The missing and the removed record each
        # break the line, i.e. take out the two segments next to them.
        expected = 287 - 2 - 2
        line = self._figure(m.showplot_outlier_detection_cleaned).axes[0].get_lines()[0]
        self.assertEqual(self._coarse_segments(line), expected)

        fig = self._figure(m.showplot_outlier_detection_qcf_timeseries)
        lines = {ln.get_label(): ln for ax in fig.axes for ln in ax.get_lines()}
        qcf = m.outlier_detection_qcf[FIELD]
        self.assertEqual(self._coarse_segments(lines[qcf.filteredseriescol]), expected)
        # The unfiltered series still holds the removed record.
        self.assertEqual(self._coarse_segments(lines[FIELD]), 287 - 2)

    def test_daily_correlation_days_draw_coarse_records(self):
        m = self._screened()
        daily = m.analysis_potential_radiation_correlation(utc_offset=1, showplot=False)[FIELD]
        fig = self._figure(lambda: m.analysis_potential_radiation_correlation(utc_offset=1))
        coarse_days = {'2020-06-01', '2020-06-02'}
        shown = set()
        for ax in fig.axes:
            day = ax.get_title().split(',')[0]
            if day in coarse_days:
                shown.add(day)
                line = [ln for ln in ax.get_lines() if ln.get_label() != 'SW_IN_POT'][0]
                self.assertGreater(_segments(line.get_ydata()), 100)
                # The plot shows the correlation the method returns.
                self.assertIn(f"r = {daily[day]:.3f}", ax.get_title())
        self.assertEqual(shown, coarse_days)

    def test_heatmaps_fill_the_period_of_coarse_records(self):
        m = self._screened()
        # 2 coarse days of 1440 one-minute cells, less the missing record (10) and
        # the 9 cells before the first record's END, which lie before the grid.
        before = 2 * 1440 - 10 - 9
        fig = self._figure(m.showplot_outlier_detection_qcf_heatmaps)
        cells = [ax.collections[0].get_array() for ax in fig.axes[:4]]
        coarse = [int(np.ma.count(c[:2])) for c in cells]
        # Panels: before QC, after QC (also without the removed record),
        # flag sum and QCF (a missing record has flags too).
        self.assertEqual(coarse, [before, before - 10, before + 10, before + 10])

        fig = self._figure(m.showplot_resampled)
        heatmaps = [ax for ax in fig.axes if ax.get_title() == f"{FIELD} (min)"]
        self.assertEqual(int(np.ma.count(heatmaps[0].collections[0].get_array()[:2])), before)

    def test_display_copies_leave_the_data_alone(self):
        m = self._screened()
        stored = {
            'orig': m.series_hires_orig[FIELD].copy(),
            'cleaned': m.series_hires_cleaned[FIELD].copy(),
            'sod': m.outlier_detection[FIELD].series_hires_cleaned.copy(),
            'detailed': m.data_detailed[FIELD].copy(),
            'flags': m.outlier_detection_qcf[FIELD].flags.copy(),
            'resampled': m.resampled_detailed[FIELD].copy(),
        }
        corr = m.analysis_potential_radiation_correlation(utc_offset=1, showplot=False)[FIELD]
        for call in (m.showplot_outlier_detection_cleaned, m.showplot_outlier_detection_qcf_heatmaps,
                     m.showplot_outlier_detection_qcf_timeseries, m.showplot_resampled):
            self._figure(call)
        shown = self._figure(lambda: m.analysis_potential_radiation_correlation(utc_offset=1))
        self.assertIsNotNone(shown)
        pd.testing.assert_series_equal(
            m.analysis_potential_radiation_correlation(utc_offset=1, showplot=False)[FIELD], corr)

        pd.testing.assert_series_equal(m.series_hires_orig[FIELD], stored['orig'])
        pd.testing.assert_series_equal(m.series_hires_cleaned[FIELD], stored['cleaned'])
        pd.testing.assert_series_equal(m.outlier_detection[FIELD].series_hires_cleaned, stored['sod'])
        pd.testing.assert_frame_equal(m.data_detailed[FIELD], stored['detailed'])
        pd.testing.assert_frame_equal(m.outlier_detection_qcf[FIELD].flags, stored['flags'])
        pd.testing.assert_frame_equal(m.resampled_detailed[FIELD], stored['resampled'])

        # The heatmap copy is a new object; writing to it reaches nothing stored.
        series = m.series_hires_orig[FIELD]
        spread = m._display_heatmap(FIELD, series)
        self.assertIsNot(spread, series)
        spread.iloc[:] = -1
        pd.testing.assert_series_equal(m.series_hires_orig[FIELD], stored['orig'])

    def test_single_resolution_draws_the_data_as_before(self):
        m = self._screened(mixed=False)
        series = m.outlier_detection[FIELD].series_hires_cleaned
        flags = m.outlier_detection_qcf[FIELD].flags
        for helper in (m._display_lines, m._display_heatmap):
            self.assertIs(helper(FIELD, series), series)
            self.assertIs(helper(FIELD, flags), flags)

        line = self._figure(m.showplot_outlier_detection_cleaned).axes[0].get_lines()[0]
        np.testing.assert_array_equal(np.asarray(line.get_ydata(), dtype=float), series.to_numpy())
        fig = self._figure(m.showplot_outlier_detection_qcf_heatmaps)
        self.assertEqual(int(np.ma.count(fig.axes[0].collections[0].get_array())),
                         int(flags[FIELD].notna().sum()))


class TestTooFewRecords(unittest.TestCase):

    def test_irregular_tiny_input_raises_a_clear_error(self):
        # Four records in two runs of two: frequency detection keeps none of them.
        index = pd.DatetimeIndex(['2024-07-01 00:10', '2024-07-01 00:20',
                                  '2024-07-01 03:21', '2024-07-01 03:22'])
        with self.assertRaisesRegex(ValueError, 'too few regular records'):
            _screening(_series_detailed(index, [1.0, 2.0, 3.0, 4.0]))


def _mixed_radiation():
    """Two days of 10MIN, then two days of 1MIN radiation-like records (nighttime offset -3)."""
    index = (pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
             .append(pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')))
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    values = np.clip(800 * np.sin(np.pi * (hour - 5) / 15), 0, None) - 3
    values += np.random.default_rng(3).normal(0, .5, len(index))
    return _series_detailed(index, values)


class TestSetCorrections(unittest.TestCase):
    """set_corrections() leaves exactly the given corrections applied, at any stage."""

    MAX = {'key': 'setto_max', 'kwargs': {'threshold': 30.5}}
    MIN = {'key': 'setto_min', 'kwargs': {'threshold': 29.5}}

    @staticmethod
    def _state(m) -> list:
        """Everything a correction changes: current series, pre-QCF series, detector series."""
        state = [m.series_hires_cleaned[FIELD].copy()]
        if FIELD in m._series_hires_corrected:
            state.append(m._series_hires_corrected[FIELD].copy())
        if FIELD in m.outlier_detection:
            state.append(m.outlier_detection[FIELD].series_hires_cleaned.copy())
        return state

    def _assert_same(self, a, b):
        sa, sb = self._state(a), self._state(b)
        self.assertEqual(len(sa), len(sb))
        for x, y in zip(sa, sb):
            pd.testing.assert_series_equal(x, y)

    @staticmethod
    def _soft(m, n_soft: int = 50):
        """Start and add a soft flag on the first records, which a strict QCF removes."""
        m.start_outlier_detection()
        sod = m.outlier_detection[FIELD]
        soft = pd.Series(0.0, index=sod.flags.index, name=f'FLAG_{FIELD}_SOFT_TEST')
        soft.iloc[:n_soft] = 1
        sod.set_pending_flag(soft)
        m.addflag()

    def test_same_list_twice_equals_once_and_the_methods(self):
        df, _ = _spiky()
        once, twice, direct = _screening(df), _screening(df), _screening(df)
        once.set_corrections([self.MAX, self.MIN])
        twice.set_corrections([self.MAX, self.MIN])
        twice.set_corrections([self.MAX, self.MIN])
        direct.correction_setto_max_threshold(threshold=30.5, showplot=False)
        direct.correction_setto_min_threshold(threshold=29.5, showplot=False)
        self._assert_same(once, twice)
        self._assert_same(once, direct)
        self.assertEqual(once.series_hires_cleaned[FIELD].max(), 30.5)

    def test_shorter_list_removes_a_correction(self):
        df, _ = _spiky()
        m, expected = _screening(df), _screening(df)
        m.set_corrections([self.MAX, self.MIN])
        m.set_corrections([self.MAX])
        expected.set_corrections([self.MAX])
        self._assert_same(m, expected)
        self.assertLess(m.series_hires_cleaned[FIELD].min(), 29.5)

    def test_replaces_earlier_correction_methods(self):
        df, _ = _spiky()
        m = _screening(df)
        m.correction_setto_max_threshold(threshold=30, showplot=False)
        m.set_corrections([])
        pd.testing.assert_series_equal(m.series_hires_cleaned[FIELD], m.series_hires_orig[FIELD])

    def test_before_start(self):
        df, _ = _spiky()
        m, expected = _screening(df), _screening(df)
        m.set_corrections([self.MIN])
        m.set_corrections([self.MAX])
        m.start_outlier_detection()
        expected.correction_setto_max_threshold(threshold=30.5, showplot=False)
        expected.start_outlier_detection()
        self._assert_same(m, expected)
        self.assertEqual(m.outlier_detection[FIELD].series_hires_cleaned.max(), 30.5)

    def test_between_tests(self):
        df, _ = _spiky()
        m, expected = _screening(df), _screening(df)
        for s in (m, expected):
            s.start_outlier_detection()
            s.flag_outliers_abslim_test(minval=0, maxval=50)
            s.addflag()  # removes the spike
        m.set_corrections([self.MIN])
        m.set_corrections([self.MAX])
        expected.correction_setto_max_threshold(threshold=30.5, showplot=False)
        self._assert_same(m, expected)
        sod_series = m.outlier_detection[FIELD].series_hires_cleaned
        self.assertEqual(int(sod_series.isna().sum()), 1)  # the added flag keeps the spike removed
        self.assertLess(sod_series.min(), 29.5)  # the replaced correction is gone

    def test_after_finalize_and_refinalize(self):
        df, _ = _spiky()
        m, expected = _screening(df), _screening(df)
        for s in (m, expected):
            self._soft(s)
            s.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        m.set_corrections([self.MIN])
        m.set_corrections([self.MAX])
        expected.correction_setto_max_threshold(threshold=30.5, showplot=False)
        self._assert_same(m, expected)
        # The QCF removals still hold.
        self.assertEqual(int(m.series_hires_cleaned[FIELD].iloc[:50].isna().sum()), 50)

        for s in (m, expected):
            s.finalize_outlier_detection()
        self._assert_same(m, expected)
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(int(cleaned.isna().sum()), 0)  # the looser run brings all back...
        self.assertEqual(cleaned.max(), 30.5)  # ...corrected
        self.assertLess(cleaned.min(), 29.5)

    def test_correction_before_finalize_undone_after(self):
        df, _ = _spiky()
        m = _screening(df)
        m.set_corrections([self.MAX])
        self._soft(m)
        m.finalize_outlier_detection(daytime_accept_qcf_below=1, nighttime_accept_qcf_below=1)
        m.set_corrections([])
        orig = m.series_hires_orig[FIELD]
        cleaned = m.series_hires_cleaned[FIELD]
        pd.testing.assert_series_equal(cleaned.iloc[50:], orig.iloc[50:], check_names=False)
        self.assertEqual(int(cleaned.iloc[:50].isna().sum()), 50)
        m.finalize_outlier_detection()
        pd.testing.assert_series_equal(m.series_hires_cleaned[FIELD], orig, check_names=False)

    def test_each_key_matches_its_method(self):
        df = _mixed_radiation()
        cases = [
            ({'key': 'radiation_zero_offset', 'kwargs': {}},
             lambda s: s.correction_remove_nighttime_zero_offset(showplot=False)),
            ({'key': 'radiation_zero_offset', 'kwargs': {'clamp_negatives': False}},
             lambda s: s.correction_remove_nighttime_zero_offset(showplot=False, clamp_negatives=False)),
            ({'key': 'relativehumidity_offset'},
             lambda s: s.correction_remove_relativehumidity_offset(showplot=False)),
            ({'key': 'setto_max', 'kwargs': {'threshold': 500}},
             lambda s: s.correction_setto_max_threshold(threshold=500, showplot=False)),
            ({'key': 'setto_min', 'kwargs': {'threshold': 0}},
             lambda s: s.correction_setto_min_threshold(threshold=0, showplot=False)),
            ({'key': 'setto_value', 'kwargs': {'dates': [['2020-06-01', '2020-06-02']], 'value': 5}},
             lambda s: s.correction_setto_value(dates=[['2020-06-01', '2020-06-02']], value=5)),
            ({'key': 'setto_value', 'kwargs': {'dates': ['2020-06-01 20:00']}},
             lambda s: s.correction_setto_value(dates=['2020-06-01 20:00'], value=0)),
            ({'key': 'set_exact_to_missing', 'kwargs': {'values': [800]}},
             lambda s: s.correction_set_exact_value_to_missing(values=[800], showplot=False)),
        ]
        orig = _screening(df)
        empty = orig.series_hires_orig[FIELD].isna()
        self.assertGreater(int(empty.sum()), 0)
        for correction, method in cases:
            with self.subTest(correction=correction):
                m, expected = _screening(df), _screening(df)
                m.set_corrections([correction])
                method(expected)
                self._assert_same(m, expected)
                self.assertEqual(int(m.series_hires_cleaned[FIELD][empty].notna().sum()), 0)

        # Dates are END timestamps: the 10MIN record ending 20:00 sits at MIDDLE 19:59:30.
        m = _screening(df)
        m.set_corrections([{'key': 'setto_value', 'kwargs': {'dates': ['2020-06-01 20:00'], 'value': -1}}])
        cleaned = m.series_hires_cleaned[FIELD]
        self.assertEqual(list(cleaned.index[cleaned == -1]), [pd.Timestamp('2020-06-01 19:59:30')])

    def test_unknown_key_raises_and_changes_nothing(self):
        df, _ = _spiky()
        m, expected = _screening(df), _screening(df)
        m.set_corrections([self.MAX])
        expected.set_corrections([self.MAX])
        with self.assertRaisesRegex(ValueError, 'Unknown correction key'):
            m.set_corrections([self.MIN, {'key': 'no_such_correction', 'kwargs': {}}])
        self._assert_same(m, expected)


class TestPublicHelpers(unittest.TestCase):
    """Display helpers, record count and resolutions for the GUI."""

    @staticmethod
    def _mixed():
        coarse = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        fine = pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')
        index = coarse.append(fine)
        return _screening(_series_detailed(index, _diurnal(index))), coarse, fine

    def test_display_helpers_return_copies(self):
        m, _, _ = self._mixed()
        series = m.series_hires_orig[FIELD]
        stored = series.copy()
        lines = m.display_lines(series)
        heatmap = m.display_heatmap(series, field=FIELD)
        pd.testing.assert_series_equal(lines, m._display_lines(FIELD, series))
        pd.testing.assert_series_equal(heatmap, m._display_heatmap(FIELD, series))
        self.assertLess(len(lines), len(series))  # empty slots inside coarse records left out
        self.assertTrue(heatmap.notna().all())  # coarse records fill their whole period
        lines.iloc[:] = -1
        heatmap.iloc[:] = -1
        pd.testing.assert_series_equal(m.series_hires_orig[FIELD], stored)

        frame = m.data_detailed[FIELD][[FIELD, 'FREQ_AUTO_SEC']]
        self.assertEqual(len(m.display_lines(frame)), len(lines))

    def test_single_resolution_returned_unchanged(self):
        index = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        m = _screening(_series_detailed(index, _diurnal(index)))
        series = m.series_hires_orig[FIELD]
        for helper in (m.display_lines, m.display_heatmap):
            shown = helper(series)
            pd.testing.assert_series_equal(shown, series)
            self.assertIsNot(shown, series)

    def test_field_required_with_more_than_one(self):
        index = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        other = 'SWC_FF1_0.15_1'
        df_other = _series_detailed(index, _diurnal(index)).rename(columns={FIELD: other})
        df_other['varname'] = other
        m = StepwiseMeteoScreeningDb(site='ch-lae', fields=[FIELD, other],
                                     data_detailed={FIELD: _series_detailed(index, _diurnal(index)),
                                                    other: df_other},
                                     site_lat=47.478333, site_lon=8.364389, utc_offset=1)
        with self.assertRaisesRegex(ValueError, 'pass field='):
            m.records_count()
        self.assertEqual(m.records_count(other), len(index))
        self.assertEqual(m.resolutions(field=other), ['10min'])

    def test_records_and_resolutions_single(self):
        index = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        values = _diurnal(index)
        values[5] = np.nan  # a record without a value is still a record
        m = _screening(_series_detailed(index.drop(index[20]), np.delete(values, 20)))
        self.assertEqual(m.records_count(), len(index) - 1)
        self.assertEqual(m.resolutions(), ['10min'])

    def test_records_and_resolutions_mixed(self):
        m, coarse, fine = self._mixed()
        self.assertEqual(len(m.data_detailed[FIELD]), 10 * len(coarse) - 9 + len(fine))
        self.assertEqual(m.records_count(), len(coarse) + len(fine))
        self.assertEqual(m.resolutions(), ['10min', '1min'])

        index = (pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
                 .append(pd.date_range('2020-06-03 00:15', '2020-06-05 00:00', freq='15min')))
        m = _screening(_series_detailed(index, _diurnal(index)))
        self.assertEqual(m.records_count(), len(index))
        self.assertEqual(m.resolutions(), ['10min', '15min'])

    def test_multi_valued_tags_are_joined(self):
        m, _, _ = self._mixed()
        self.assertEqual(m.tags[FIELD]['freq'], TAGS['freq'])
        coarse = pd.date_range('2020-06-01 00:10', '2020-06-03 00:00', freq='10min')
        fine = pd.date_range('2020-06-03 00:01', '2020-06-05 00:00', freq='1min')
        df = _series_detailed(coarse.append(fine), _diurnal(coarse.append(fine)))
        df.loc[fine, 'freq'] = '1min'
        self.assertEqual(_screening(df).tags[FIELD]['freq'], '10min,1min')


class TestCodegenScriptRuns(unittest.TestCase):
    """The script from ``meteoscreening_to_code`` runs and gives what the same
    calls on the class give."""

    STEPS = [{'method': 'flag_outliers_abslim_test', 'kwargs': {'minval': 0, 'maxval': 50}},
             {'method': 'flag_outliers_zscore_test', 'kwargs': {'thres_zscore': 5}}]
    CORRECTIONS = [{'key': 'setto_max', 'kwargs': {'threshold': 31}}]
    COORDS = dict(site='ch-lae', site_lat=47.478333, site_lon=8.364389, utc_offset=1)

    def _code(self, download=None):
        from diive.preprocessing.qaqc.codegen import meteoscreening_to_code
        return meteoscreening_to_code(self.STEPS, field=FIELD, **self.COORDS, download=download,
                                      corrections=self.CORRECTIONS, to_freqstr='30min',
                                      agg='mean', mincounts_perc=.5)

    def _expected(self, df) -> pd.DataFrame:
        m = StepwiseMeteoScreeningDb(data_detailed={FIELD: df}, fields=FIELD, **self.COORDS)
        m.start_outlier_detection()
        for step in self.STEPS:
            getattr(m, step['method'])(**step['kwargs'])
            m.addflag()
        m.finalize_outlier_detection()
        m.set_corrections(self.CORRECTIONS)
        m.resample(to_freqstr='30min', agg='mean', mincounts_perc=.5)
        return m.resampled_detailed[FIELD]

    def test_with_the_download_replaced_by_data(self):
        df, spike = _spiky()
        code = self._code()
        placeholder = 'data_detailed = ...\n'
        self.assertIn(placeholder, code)
        namespace = {'data_detailed': {FIELD: df}}
        exec(compile(code.replace(placeholder, ''), '<script>', 'exec'), namespace)
        resampled = namespace['resampled']
        pd.testing.assert_frame_equal(resampled, self._expected(df))
        self.assertLessEqual(resampled[FIELD].max(), 31)
        # The spike is removed (the screening grid is TIMESTAMP_MIDDLE).
        cleaned = namespace['mscr'].series_hires_cleaned[FIELD]
        self.assertTrue(np.isnan(cleaned[spike - pd.Timedelta('5min')]))

    def test_download_call_as_rendered(self):
        df, _ = _spiky()
        calls = {}

        class FakeInfluxIO:
            def __init__(self, dirconf):
                calls['dirconf'] = dirconf

            def download(self, **kwargs):
                calls['download'] = kwargs
                return None, {FIELD: df}, None

        download = {'bucket': 'ch-lae_raw', 'measurement': 'SWC', 'data_version': 'raw',
                    'start': '2020-06-01 00:10:00', 'stop': '2020-06-04 00:10:00',
                    'dirconf': 'configs'}
        namespace = {}
        with mock.patch('diive.core.io.db.influx.InfluxIO', FakeInfluxIO):
            exec(compile(self._code(download), '<script>', 'exec'), namespace)
        self.assertEqual(calls['dirconf'], 'configs')
        self.assertEqual(calls['download'], dict(
            bucket='ch-lae_raw', measurements=['SWC'], fields=[FIELD],
            start='2020-06-01 00:10:00', stop='2020-06-04 00:10:00',
            timezone_offset_to_utc_hours=1, data_version='raw'))
        pd.testing.assert_frame_equal(namespace['resampled'], self._expected(df))


if __name__ == '__main__':
    unittest.main()
