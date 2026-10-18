"""Faster code paths give the same results as the formulas they replaced (finding L284)."""
import contextlib
import io
import threading
import unittest

import numpy as np
import pandas as pd

from diive.core.base.flagbase import FlagBase, sum_flag_value
from diive.variables import radiation
from diive.variables.radiation import potrad


def _random_flags(seed: int, n_rows: int = 500, n_cols: int = 6, nan_frac: float = 0.2) -> pd.DataFrame:
    """Flag frame with values 0, 1, 2 and NaN, including some all-NaN rows."""
    rng = np.random.default_rng(seed)
    vals = rng.choice([0.0, 1.0, 2.0], size=(n_rows, n_cols))
    vals[rng.random((n_rows, n_cols)) < nan_frac] = np.nan
    vals[:5] = np.nan
    idx = pd.date_range('2024-01-01', periods=n_rows, freq='10min', name='TIMESTAMP_END')
    return pd.DataFrame(vals, index=idx, columns=[f'FLAG_TA_T{i}_TEST' for i in range(n_cols)])


class TestSumFlagValue(unittest.TestCase):

    def _assert_like_old(self, df, value):
        old = df[df == value].sum(axis=1)
        new = sum_flag_value(df, value)
        pd.testing.assert_series_equal(new, old, check_exact=True, check_freq=True)

    def test_float_frames_match_masked_sum(self):
        for seed in range(10):
            for n_cols in (1, 3, 8):
                df = _random_flags(seed, n_cols=n_cols)
                for value in (1, 2):
                    with self.subTest(seed=seed, n_cols=n_cols, value=value):
                        self._assert_like_old(df, value)

    def test_other_dtypes_keep_old_result(self):
        # int, float32, mixed and column-less frames take the pandas formula
        idx = pd.date_range('2024-01-01', periods=4, freq='h')
        frames = [
            pd.DataFrame({'a': [2, 2, 2, 2], 'b': [2, 2, 2, 2]}, index=idx),
            pd.DataFrame({'a': [2, 0, 1, 2]}, index=idx, dtype='float32'),
            pd.DataFrame({'a': [2, 0, 1, 2], 'b': [2.0, np.nan, 2.0, 1.0]}, index=idx),
            pd.DataFrame(index=idx),
        ]
        for i, df in enumerate(frames):
            with self.subTest(frame=i):
                self._assert_like_old(df, 2)


class TestQCFFlagsums(unittest.TestCase):

    def test_flagsums_match_old_formula(self):
        from diive.qaqc import FlagQCF
        for seed in range(5):
            df = _random_flags(seed)
            df['TA'] = np.random.default_rng(seed).normal(size=len(df))
            qcf = FlagQCF(df=df, target_col='TA')
            new = qcf._calculate_flagsums(df.copy())
            tests = df[[c for c in df.columns if c.endswith('_TEST')]]
            hard = tests[tests == 2].sum(axis=1)
            soft = tests[tests == 1].sum(axis=1)
            with self.subTest(seed=seed):
                pd.testing.assert_series_equal(new[qcf.sumhardflagscol], hard.rename(qcf.sumhardflagscol),
                                               check_exact=True)
                pd.testing.assert_series_equal(new[qcf.sumsoftflagscol], soft.rename(qcf.sumsoftflagscol),
                                               check_exact=True)
                pd.testing.assert_series_equal(new[qcf.sumflagscol], hard.add(soft).rename(qcf.sumflagscol),
                                               check_exact=True)


class TestRepeatOverallFlag(unittest.TestCase):

    def test_overall_flag_matches_old_formula(self):
        for seed in range(5):
            frames = _random_flags(seed, n_cols=4)
            series = pd.Series(np.random.default_rng(seed).normal(size=len(frames)), index=frames.index,
                               name='TA')
            series.iloc[::7] = np.nan
            flagbase = FlagBase(series=series.asfreq('10min'), flagid='OUTLIER_TEST', verbose=False)
            iterations = [frames.iloc[:, [i]] for i in range(frames.shape[1])]

            def func(iteration):
                # Outliers in all but the last iteration, so the loop runs through all
                return iterations[iteration - 1], (1 if iteration < len(iterations) else 0)

            overall, n_iterations = flagbase.repeat(func, repeat=True)
            old = frames[frames == 2].sum(axis=1).where(flagbase.series.notna())
            with self.subTest(seed=seed):
                self.assertEqual(n_iterations, frames.shape[1])
                pd.testing.assert_series_equal(overall, old.rename(flagbase.generate_flagname()),
                                               check_exact=True)


class TestReportQCFFlags(unittest.TestCase):

    @staticmethod
    def _old_report(qcf):
        """report_qcf_flags as it was before L284, filtering the records per column."""
        from diive.core.base.identify import identify_flagcols
        from diive.preprocessing.qaqc import qcf as qcf_mod
        out = qcf_mod._console.print

        def flagstats_dt_nt(col, df):
            test_name = col.replace('FLAG_', '').replace('_TEST', '')
            out(f"\n  +- {test_name}")
            out(f"  |  {'Period':<15} | {'Pass (0)':<15} | {'Warn (1)':<15} | {'Fail (2)':<15} | {'Missing':<12}")
            out(f"  |  {'-' * 85}")
            qcf._flagstats(flag=df[col], prefix="OVERALL", indent="  |  ")
            for prefix, mask in (("DAYTIME", qcf.daytime), ("NIGHTTIME", qcf.nighttime)):
                if isinstance(mask, pd.Series):
                    flag = df[col].loc[mask == 1]
                    if len(flag) > 0:
                        qcf._flagstats(flag=flag, prefix=prefix, indent="  |  ")

        out(f"\n\n{'=' * 100}")
        out(f"  INDIVIDUAL TEST FLAG STATISTICS: {qcf.series_name}")
        out(f"{'=' * 100}")
        flagcols = identify_flagcols(df=qcf.flags, seriescol=qcf.series_name)
        test_flagcols = [c for c in flagcols if str(c).endswith('_TEST')]
        out(f"\n  Total test flags: {len(test_flagcols)}")
        out(f"\n\n  +- REPORT 1A: ALL RECORDS (INCLUDING MISSING VALUES)")
        out(f"  |")
        for col in test_flagcols:
            flagstats_dt_nt(col, qcf.flags[qcf._is_record])
        out(f"\n  +- REPORT 1B: AVAILABLE RECORDS ONLY (EXCLUDING MISSING VALUES)")
        out(f"  |")
        _df = qcf.flags[qcf._is_record].copy()
        _df = _df[~_df[qcf.series_name].isnull()].copy()
        for col in test_flagcols:
            flagstats_dt_nt(col, _df)
        out(f"\n{'=' * 100}\n")

    def test_printed_tables_match_old_loop(self):
        from diive.qaqc import FlagQCF
        for with_swinpot in (False, True):
            df = _random_flags(3, n_rows=48 * 6)
            df['TA'] = np.random.default_rng(3).normal(size=len(df))
            df.loc[df.index[::5], 'TA'] = np.nan
            kwargs = {}
            if with_swinpot:
                df['SW_IN_POT'] = potrad(df.index, lat=47.29, lon=7.73, utc_offset=1).to_numpy()
                kwargs = dict(swinpot_col='SW_IN_POT', nighttime_threshold=20)
            qcf = FlagQCF(df=df, target_col='TA', **kwargs)
            qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
            new_buf, old_buf = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(new_buf):
                qcf.report_qcf_flags()
            with contextlib.redirect_stdout(old_buf):
                self._old_report(qcf)
            with self.subTest(with_swinpot=with_swinpot):
                self.assertEqual(new_buf.getvalue(), old_buf.getvalue())
                self.assertEqual('NIGHTTIME' in new_buf.getvalue(), with_swinpot)


class TestPotradCache(unittest.TestCase):
    """The cache returns what `_oneflux_window_mean` computes, for every input."""

    LAT, LON, OFF = 47.29, 7.73, 1

    def setUp(self):
        radiation._WINDOW_MEAN_CACHE.clear()

    def _uncached(self, idx, lat=LAT, lon=LON, off=OFF, period_min=30):
        return radiation._oneflux_window_mean(pd.DatetimeIndex(idx), lat, lon, off, period_min)

    def _idx(self, start='2024-03-01 00:15', periods=2000, tz=None):
        return pd.date_range(start, periods=periods, freq='30min', tz=tz)

    def test_equal_inputs_equal_arrays_and_copies(self):
        first = potrad(self._idx(), lat=self.LAT, lon=self.LON, utc_offset=self.OFF)
        first.iloc[:] = -1.0  # a caller changing its result must not change the cache
        second = potrad(self._idx(), lat=self.LAT, lon=self.LON, utc_offset=self.OFF)
        self.assertEqual(len(radiation._WINDOW_MEAN_CACHE), 1)
        np.testing.assert_array_equal(second.to_numpy(), self._uncached(self._idx()))

    def test_different_inputs_different_arrays(self):
        base = self._idx()
        inputs = [
            dict(idx=base),
            dict(idx=self._idx(start='2024-03-02 00:15')),  # same length, shifted a day
            dict(idx=base.tz_localize('UTC')),  # same int64 values, different dtype
            dict(idx=base.tz_localize('UTC').tz_convert('Etc/GMT-1')),  # same instants, other wall time
            dict(idx=base, lat=-self.LAT),
            dict(idx=base, lon=-self.LON),
            dict(idx=base, off=2),
            dict(idx=base, lat=np.float32(self.LAT)),
        ]
        results = []
        for args in inputs:
            idx = args['idx']
            lat, lon, off = args.get('lat', self.LAT), args.get('lon', self.LON), args.get('off', self.OFF)
            got = potrad(idx, lat=lat, lon=lon, utc_offset=off).to_numpy()
            with self.subTest(args={k: v for k, v in args.items() if k != 'idx'}, tz=idx.tz):
                np.testing.assert_array_equal(got, self._uncached(idx, lat, lon, off))
            results.append(got)
        # Each call above was a separate entry (the cache holds the last four)
        self.assertEqual(len(radiation._WINDOW_MEAN_CACHE), radiation._WINDOW_MEAN_CACHE_SIZE)
        self.assertFalse(np.array_equal(results[0], results[1]))
        self.assertFalse(np.array_equal(results[0], results[3]))
        # Repeating the first input after it was evicted still gives its own values
        again = potrad(base, lat=self.LAT, lon=self.LON, utc_offset=self.OFF).to_numpy()
        np.testing.assert_array_equal(again, results[0])

    def test_short_index_not_cached(self):
        idx = self._idx(periods=10)
        got = potrad(idx, lat=self.LAT, lon=self.LON, utc_offset=self.OFF).to_numpy()
        np.testing.assert_array_equal(got, self._uncached(idx))
        self.assertEqual(len(radiation._WINDOW_MEAN_CACHE), 0)

    def test_threads_get_correct_values(self):
        indexes = [self._idx(start=f'2024-0{m}-01 00:15') for m in range(1, 7)]
        expected = [self._uncached(idx) for idx in indexes]
        errors = []

        def work(k):
            for _ in range(3):
                i = (k + _) % len(indexes)
                got = potrad(indexes[i], lat=self.LAT, lon=self.LON, utc_offset=self.OFF).to_numpy()
                if not np.array_equal(got, expected[i]):
                    errors.append(i)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(radiation._WINDOW_MEAN_CACHE), radiation._WINDOW_MEAN_CACHE_SIZE)


if __name__ == '__main__':
    unittest.main()
