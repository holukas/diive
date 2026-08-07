"""
Tests for UstarMovingPointDetection (ONEFlux moving-point USTAR detection).

The key guard is `test_vectorized_matches_literal_reference`, which compares the
vectorized numpy core against an independent, line-by-line transcription of the
reference C loops (oneflux_steps/ustar_mp/src/ustar.c). Any change to the core
that diverges from the C algorithm will break it.
"""

import unittest

import numpy as np

from diive.configs.exampledata import load_exampledata_parquet_lae
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection


# --- independent literal transcription of ustar.c (no vectorization shortcuts) ---

NF = UstarMovingPointDetection.THRESHOLD_NOT_FOUND
WS = UstarMovingPointDetection.WINDOW_SIZE_FORWARD_MODE
TC = UstarMovingPointDetection.THRESHOLD_CHECK
MINTA = UstarMovingPointDetection.MIN_SAMPLES_TA_CLASS
CORR = UstarMovingPointDetection.CORRELATION_CHECK
FU = UstarMovingPointDetection.FIRST_USTAR_MEAN_CHECK


def _meanws(arr, index, ec):
    n = len(arr)
    if index > n:
        return 0.0
    s, c = 0.0, 0
    for i in range(index, index + ec):
        if i >= n:
            break
        s += arr[i]
        c += 1
    return float("nan") if c == 0 else s / c


def _fwd(um, fm, n):
    nc = len(um)
    if n < 1 or nc - n <= 0:
        return NF
    for i in range(0, nc - n + 1):
        means, bad = [], False
        for y in range(n):
            m = _meanws(fm, i + 1 + y, WS)
            if not np.isfinite(m):
                bad = True
                break
            means.append(m)
        if bad:
            continue
        if sum(1 for y in range(n) if fm[i + y] >= means[y] * TC) == n:
            return um[i]
    return NF


def _bounds(v, ncl, npc):
    N = len(v)
    B = [(-1, -1)] * ncl
    ce, broke = 0, False
    for i in range(ncl - 1):
        cs = ce
        ce = npc * (i + 1) - 1
        if cs >= N:
            broke = True
            break
        if ce >= N:
            ce = N - 1
        val = v[ce]
        ce += 1
        while ce < N and v[ce] == val:
            ce += 1
        B[i] = (cs, ce - 1)
    if not broke and ce < N:
        B[ncl - 1] = (ce, N - 1)
    return B


def _means_of(arr, B):
    out = [0.0] * len(B)
    for k, (s, e) in enumerate(B):
        if s >= 0:
            out[k] = arr[s:e + 1].mean()
    return out


def _pear(x, y):
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    dn = np.sqrt((dx * dx).sum()) * np.sqrt((dy * dy).sum())
    return float("nan") if dn == 0 else (dx * dy).sum() / dn


def _det_ta(nc, uc, uclasses, fn):
    m = len(uc)
    npu = m // uclasses
    if npu < 1:
        return NF
    o = np.argsort(uc, kind="stable")
    us, ns = uc[o], nc[o]
    B = _bounds(us, uclasses, npu)
    um, fm = _means_of(us, B), _means_of(ns, B)
    if um[0] > FU:
        return NF
    return _fwd(um, fm, fn)


def _det_season(nee, ta, ustar, taclasses, uclasses, fn):
    N = len(nee)
    npt = N // taclasses
    if npt < MINTA:
        return NF
    o = np.argsort(ta, kind="stable")
    ts, us, ns = ta[o], ustar[o], nee[o]
    B = _bounds(ts, taclasses, npt)
    ths = []
    for (s, e) in B:
        if s < 0 or (e - s + 1) < MINTA:
            continue
        c = _pear(ts[s:e + 1], us[s:e + 1])
        if not np.isfinite(c) or abs(c) > CORR:
            continue
        t = _det_ta(ns[s:e + 1], us[s:e + 1], uclasses, fn)
        if np.isfinite(t) and t != NF:
            ths.append(t)
    return float(np.median(ths)) if ths else NF


class TestUstarMovingPointDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_exampledata_parquet_lae()

    def test_detect_runs_and_is_plausible(self):
        d = UstarMovingPointDetection(self.df, verbose=0)
        res = d.detect()
        self.assertEqual(len(res), 4)
        annual = d.get_annual_thresholds()["threshold"]
        # plausible USTAR threshold range for a forest site
        self.assertGreater(annual, 0.0)
        self.assertLess(annual, 2.0)

    def test_default_seasons_are_calendar_quarters(self):
        d = UstarMovingPointDetection(self.df)
        self.assertEqual(d.season_groups, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    def test_annual_is_max_across_seasons(self):
        d = UstarMovingPointDetection(self.df)
        res = d.detect()
        vals = res["threshold"].to_numpy()
        valid = vals[np.isfinite(vals) & (vals != d.THRESHOLD_NOT_FOUND)]
        self.assertAlmostEqual(
            d.get_annual_thresholds()["threshold"], float(np.max(valid)), places=9
        )

    def test_vectorized_matches_literal_reference(self):
        """The numpy core must reproduce the literal C transcription exactly."""
        for taclasses, uclasses, fn in [(7, 20, 2), (7, 20, 1), (5, 15, 2), (7, 20, 3)]:
            d = UstarMovingPointDetection(
                self.df, ta_classes_count=taclasses,
                ustar_classes_count=uclasses, forward_mode_n=fn,
            )
            nee, ta, ustar, month, valid, night = d._night_valid_arrays()
            nee_n, ta_n, ustar_n, month_n = nee[night], ta[night], ustar[night], month[night]

            ref = [
                _det_season(
                    nee_n[np.isin(month_n, months)],
                    ta_n[np.isin(month_n, months)],
                    ustar_n[np.isin(month_n, months)],
                    taclasses, uclasses, fn,
                )
                for months in d.season_groups
            ]
            ref = [np.nan if r == NF else r for r in ref]
            got = d.detect()["threshold"].to_numpy()
            np.testing.assert_allclose(
                got, ref, equal_nan=True,
                err_msg=f"mismatch for ta={taclasses}, ustar={uclasses}, fw={fn}",
            )

    def test_bootstrap_has_annual_row(self):
        d = UstarMovingPointDetection(self.df, verbose=0)
        bs = d.bootstrap(n_iter=15)
        self.assertIn("Annual", bs.index)
        self.assertEqual(len(bs), 5)  # 4 seasons + annual
        self.assertTrue(set(["mean", "std", "p05", "p50", "p95"]).issubset(bs.columns))

    def test_forward_mode_n_validation(self):
        with self.assertRaises(ValueError):
            UstarMovingPointDetection(self.df, forward_mode_n=0)


class TestUstarBootstrapVutCut(unittest.TestCase):
    """VUT (per-year) and CUT (constant) accessors on the bootstrap wrapper."""

    @classmethod
    def setUpClass(cls):
        from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
        df = load_exampledata_parquet_lae()
        cls.boot = UstarBootstrapThresholds(
            df, detector_class=UstarMovingPointDetection,
            n_iter=10, percentiles=(16, 50, 84), n_jobs=1, verbose=0)
        cls.boot.run()

    def test_vut_is_per_year_table(self):
        vut = self.boot.get_vut_thresholds()
        # One row per calendar year; percentile columns.
        self.assertEqual(list(vut.index), self.boot.years_)
        self.assertEqual(list(vut.columns), ["p16", "p50", "p84"])
        # get_vut_thresholds() returns the same object as run() / annual_stats_.
        self.assertTrue(vut.equals(self.boot.annual_stats_))

    def test_cut_is_constant_dict(self):
        cut = self.boot.get_cut_threshold()
        self.assertEqual(set(cut), {"p16", "p50", "p84"})
        # Percentiles are ordered: p16 <= p50 <= p84.
        self.assertLessEqual(cut["p16"], cut["p50"])
        self.assertLessEqual(cut["p50"], cut["p84"])

    def test_vut_before_run_raises(self):
        from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
        b = UstarBootstrapThresholds(
            load_exampledata_parquet_lae(),
            detector_class=UstarMovingPointDetection, n_iter=5)
        with self.assertRaises(RuntimeError):
            b.get_vut_thresholds()


if __name__ == "__main__":
    unittest.main()


class TestRecordMinimumIsEnforcedEverywhere(unittest.TestCase):
    """The record minimum must gate every entry point, not just detect().

    The u* threshold decides which nighttime fluxes are discarded, so a threshold
    from a record the detector itself considers too short is worse than none. The
    bootstrap paths used to skip the check — and they are the ones
    UstarBootstrapThresholds and the flux chain's L3.3 detection actually call.
    """

    @staticmethod
    def _synthetic(n):
        import pandas as pd
        ix = pd.date_range('2023-01-01', periods=n, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(0)
        hr = ix.hour + ix.minute / 60
        sw = np.clip(500 * np.sin(2 * np.pi * (hr - 6) / 24), 0, None)
        return pd.DataFrame({'NEE': rng.randn(n), 'TA': 10 + 5 * rng.randn(n),
                             'USTAR': np.abs(rng.randn(n)) * 0.5, 'SW_IN': sw},
                            index=ix)

    def _detector(self, n):
        return UstarMovingPointDetection(
            df=self._synthetic(n), nee_col='NEE', ta_col='TA',
            ustar_col='USTAR', swin_col='SW_IN', verbose=0)

    def test_every_entry_point_refuses_a_short_record(self):
        n = UstarMovingPointDetection.MIN_SAMPLES_PERIOD // 3
        for name in ('detect', 'bootstrap', 'bootstrap_annual_samples'):
            with self.subTest(entry=name):
                det = self._detector(n)
                with self.assertRaises(ValueError) as ctx:
                    getattr(det, name)() if name == 'detect' else \
                        getattr(det, name)(n_iter=3)
                self.assertIn('Insufficient', str(ctx.exception))

    def test_a_sufficient_record_still_runs_everywhere(self):
        n = UstarMovingPointDetection.MIN_SAMPLES_PERIOD * 2
        self._detector(n).detect()
        self._detector(n).bootstrap(n_iter=3)
        self._detector(n).bootstrap_annual_samples(n_iter=3)


class TestBootstrapReportsWhyAWindowFailed(unittest.TestCase):
    """An empty window becomes a NaN threshold, so the reason must reach the user.

    The wrapper catches every exception from the worker; without carrying the
    reason out, 'too few records', 'mistyped column name' and 'detection found
    nothing' are indistinguishable, and all three surface as bare NaNs.
    """

    @staticmethod
    def _two_short_years():
        import pandas as pd
        frames = []
        for y in (2020, 2021):
            ix = pd.date_range(f'{y}-01-01', periods=800, freq='30min',
                               name='TIMESTAMP_MIDDLE')
            rng = np.random.RandomState(y)
            hr = ix.hour + ix.minute / 60
            sw = np.clip(500 * np.sin(2 * np.pi * (hr - 6) / 24), 0, None)
            frames.append(pd.DataFrame(
                {'NEE': rng.randn(800), 'TA': 10 + 5 * rng.randn(800),
                 'USTAR': np.abs(rng.randn(800)) * 0.5, 'SW_IN': sw}, index=ix))
        return pd.concat(frames)

    def test_the_worker_returns_the_reason(self):
        from diive.flux.lowres.ustar_bootstrap import _bootstrap_window_worker
        df = self._two_short_years()
        year, samples, err = _bootstrap_window_worker(
            2020, df, UstarMovingPointDetection,
            dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR', swin_col='SW_IN'), 3)
        self.assertEqual(samples, [])
        self.assertIsNotNone(err, "an empty window must carry its reason")
        self.assertIn('Insufficient', err)

    def test_a_typo_is_not_mistaken_for_missing_data(self):
        from diive.flux.lowres.ustar_bootstrap import _bootstrap_window_worker
        df = self._two_short_years()
        _year, samples, err = _bootstrap_window_worker(
            2020, df, UstarMovingPointDetection,
            dict(nee_col='TYPO', ta_col='TA', ustar_col='USTAR', swin_col='SW_IN'), 3)
        self.assertEqual(samples, [])
        self.assertNotIn('Insufficient', err)


class TestFailedDetectionReportsNoThreshold(unittest.TestCase):
    """A failed detection must not leave a usable-looking threshold behind.

    `THRESHOLD_NOT_FOUND` is 10.0 m/s - a plausible-looking u* threshold that would
    filter out every record. It was stored in the documented `annual_thresholds_`
    attribute and converted back to NaN only inside `get_annual_thresholds()`, so
    reading the attribute directly after a failed detection was a trap.
    """

    @staticmethod
    def _undetectable():
        # Enough records to clear the minimum, but USTAR never varies, so no TA
        # class yields a threshold and every season comes back empty.
        import pandas as pd
        n = UstarMovingPointDetection.MIN_SAMPLES_PERIOD * 2
        ix = pd.date_range('2020-01-01 00:30', periods=n, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(0)
        return pd.DataFrame({'NEE': rng.normal(0, 5, n), 'TA': rng.normal(10, 5, n),
                             'USTAR': 0.5, 'SW_IN': 0.0}, index=ix)

    def test_the_attribute_and_the_accessor_agree_on_nan(self):
        det = UstarMovingPointDetection(df=self._undetectable(), nee_col='NEE', ta_col='TA',
                                        ustar_col='USTAR', swin_col='SW_IN', verbose=0)
        seasonal = det.detect()['threshold']
        self.assertTrue(seasonal.isna().all(), 'this fixture must fail detection')
        stored = det.annual_thresholds_['threshold']
        self.assertTrue(np.isnan(stored))
        self.assertNotEqual(stored, UstarMovingPointDetection.THRESHOLD_NOT_FOUND)
        self.assertTrue(np.isnan(det.get_annual_thresholds()['threshold']))

    def test_a_real_threshold_still_arrives_intact(self):
        import pandas as pd
        n = UstarMovingPointDetection.MIN_SAMPLES_PERIOD * 2
        ix = pd.date_range('2020-01-01 00:30', periods=n, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(0)
        df = pd.DataFrame({'NEE': rng.normal(0, 5, n), 'TA': rng.normal(10, 5, n),
                           'USTAR': rng.uniform(0, 1, n), 'SW_IN': 0.0}, index=ix)
        det = UstarMovingPointDetection(df=df, nee_col='NEE', ta_col='TA',
                                        ustar_col='USTAR', swin_col='SW_IN', verbose=0)
        det.detect()
        stored = det.annual_thresholds_['threshold']
        self.assertTrue(np.isfinite(stored))
        self.assertEqual(stored, det.get_annual_thresholds()['threshold'])


class TestUstarFlagNeedsAMeasuredUstar(unittest.TestCase):
    """A record with no USTAR cannot be shown to be turbulent, so it cannot pass.

    Both comparisons are False against NaN, so such a record used to land in
    neither the accepted nor the rejected set and its flag summed to 0 - accepted,
    with turbulence unknown. Flagging it NaN would not help either: FlagQCF sums
    only 1s and 2s, so a NaN flag survives downstream just the same.
    """

    @staticmethod
    def _frame():
        import pandas as pd
        ix = pd.date_range('2023-01-01 00:30', periods=6, freq='h',
                           name='TIMESTAMP_MIDDLE')
        nee = pd.Series([3.0, np.nan, 4.0, 5.0, 6.0, 7.0], index=ix, name='NEE')
        ustar = pd.Series([np.nan, 0.5, 0.1, 0.9, np.nan, 0.4], index=ix, name='USTAR')
        return nee, ustar

    def test_a_missing_ustar_is_rejected_not_accepted(self):
        from diive.flux.lowres.ustarthreshold import FlagSingleConstantUstarThreshold
        nee, ustar = self._frame()
        flag = FlagSingleConstantUstarThreshold(series=nee, ustar=ustar, threshold=0.3).run().get_flag()
        self.assertEqual(flag.iloc[0], 2)   # flux present, USTAR unknown
        self.assertEqual(flag.iloc[4], 2)   # same
        self.assertEqual(flag.iloc[2], 2)   # measured USTAR below the threshold
        self.assertEqual(flag.iloc[3], 0)   # measured USTAR above it
        self.assertEqual(flag.iloc[5], 0)
        # A missing flux stays "not testable" (NaN), which is a separate question.
        self.assertTrue(np.isnan(flag.iloc[1]))

    def test_a_threshold_series_with_holes_is_refused(self):
        # Reindexing a threshold Series that does not span the record fills the
        # rest with NaN, which would reject those records without saying why.
        import pandas as pd
        from diive.flux.lowres.ustarthreshold import FlagMultipleVariableUstarThresholds
        nee, ustar = self._frame()
        thr = pd.Series(0.3, index=nee.index)
        thr.iloc[-2:] = np.nan
        flagger = FlagMultipleVariableUstarThresholds(
            series=nee, ustar=ustar, threshold_series={'VUT_50': thr},
            showplot=False, verbose=False)
        with self.assertRaises(ValueError) as ctx:
            flagger.calc()
        self.assertIn('VUT_50', str(ctx.exception))

    def test_a_complete_threshold_series_still_works(self):
        import pandas as pd
        from diive.flux.lowres.ustarthreshold import FlagMultipleVariableUstarThresholds
        nee, ustar = self._frame()
        thr = pd.Series(0.3, index=nee.index)
        flagger = FlagMultipleVariableUstarThresholds(
            series=nee, ustar=ustar, threshold_series={'VUT_50': thr},
            showplot=False, verbose=False)
        flagger.calc()
        flagcol = [c for c in flagger.results.columns if c.startswith('FLAG_')][0]
        self.assertEqual(flagger.results[flagcol].tolist()[3], 0)
        self.assertEqual(flagger.results[flagcol].tolist()[0], 2)
