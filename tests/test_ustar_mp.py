"""
Tests for UstarMovingPointDetection (ONEFlux moving-point USTAR detection).

The key guard is `test_kernels_match_literal_reference`, which compares the
numba kernels against an independent, line-by-line pure-Python transcription
of the reference C loops (oneflux_steps/ustar_mp/src/ustar.c), bitwise. The
full comparison against the compiled C (20 site-years, bootstrap included)
lives outside the repo in diive-data/references/ustar_parity/mp/.
"""

import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no plot windows during tests

from diive.configs.exampledata import load_exampledata_parquet_lae
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection

INVALID = -9999.0
NF = UstarMovingPointDetection.THRESHOLD_NOT_FOUND
WS = UstarMovingPointDetection.WINDOW_SIZE_FORWARD_MODE
TC = UstarMovingPointDetection.THRESHOLD_CHECK

# --- independent literal transcription of ustar.c (plain Python, no numba) ---


def _inv(v):
    return -10000.0 < v <= -9999.0  # IS_INVALID_VALUE: -9999 == (int)v


def _qsort(rows, start, num, col):
    """Microsoft UCRT qsort (qsort.cpp) on a list of row lists, by rows[i][col]."""
    def cmp(a, b):
        x, y = rows[a][col], rows[b][col]
        return -1 if x < y else (1 if x > y else 0)

    def swap(a, b):
        rows[a], rows[b] = rows[b], rows[a]

    if num < 2:
        return
    stack = []
    lo, hi = start, start + num - 1
    while True:
        if hi - lo + 1 <= 8:
            while hi > lo:  # shortsort
                mx = lo
                for p in range(lo + 1, hi + 1):
                    if cmp(p, mx) > 0:
                        mx = p
                swap(mx, hi)
                hi -= 1
        else:
            mid = lo + (hi - lo + 1) // 2
            if cmp(lo, mid) > 0:
                swap(lo, mid)
            if cmp(lo, hi) > 0:
                swap(lo, hi)
            if cmp(mid, hi) > 0:
                swap(mid, hi)
            loguy, higuy = lo, hi
            while True:
                if mid > loguy:
                    loguy += 1
                    while loguy < mid and cmp(loguy, mid) <= 0:
                        loguy += 1
                if mid <= loguy:
                    loguy += 1
                    while loguy <= hi and cmp(loguy, mid) <= 0:
                        loguy += 1
                higuy -= 1
                while higuy > mid and cmp(higuy, mid) > 0:
                    higuy -= 1
                if higuy < loguy:
                    break
                swap(loguy, higuy)
                if mid == higuy:
                    mid = loguy
            higuy += 1
            if mid < higuy:
                higuy -= 1
                while higuy > mid and cmp(higuy, mid) == 0:
                    higuy -= 1
            if mid >= higuy:
                higuy -= 1
                while higuy > lo and cmp(higuy, mid) == 0:
                    higuy -= 1
            if higuy - lo >= hi - loguy:
                if lo < higuy:
                    stack.append((lo, higuy))
                if loguy < hi:
                    lo = loguy
                    continue
            else:
                if loguy < hi:
                    stack.append((loguy, hi))
                if lo < higuy:
                    hi = higuy
                    continue
        if not stack:
            return
        lo, hi = stack.pop()


def _meanws(arr, index, ec):
    n = len(arr)
    if index > n:
        return 0.0
    s, c = 0.0, 0
    for i in range(index, index + ec):
        if i >= n:
            break
        if _inv(arr[i]):
            return INVALID
        s += arr[i]
        c += 1
    m = s / c if c else float("nan")
    return INVALID if m != m else m


def _fwd(um, fm, n):
    nc = len(um)
    if n < 1 or nc - n <= 0:
        return INVALID
    for i in range(nc - n + 1):
        means = []
        for y in range(n):
            m = _meanws(fm, i + 1 + y, WS)
            if _inv(m):
                break
            means.append(m)
        if len(means) < n:
            continue
        if sum(1 for y in range(n) if fm[i + y] >= means[y] * TC) == n:
            return um[i]
    return NF


def _windows(rows, off, n, ncl, col, start_end_branch):
    """Class windows of ustar.c:602-637 (TA) and 693-732 (USTAR), absolute indices."""
    win = [(-1, -1)] * ncl
    per = n // ncl
    end = 0
    i = 0
    while i < ncl - 1:
        start = end
        end = per * (i + 1) - 1
        if start >= n:
            break
        if end >= n:
            end = n - 1
        value = rows[off + end][col]
        end += 1
        while end < n:
            if value != rows[off + end][col]:
                if start_end_branch and start == end:
                    value = rows[off + end][col]
                    end += 1
                    continue
                break
            end += 1
        win[i] = (off + start, off + end - 1)
        i += 1
    if end < n:
        win[i] = (off + end, off + n - 1)
    return win


def _div(s, cnt):
    # C double / int: 0.0 / 0 is NaN, x / negative is fine
    return s / cnt if cnt else float("nan")


def _ustar_threshold(rows, rows_count, days, samples, nseas, tacl, ucl, fn):
    """ustar.c::ustar_threshold, forward mode fn only. Returns the container or None."""
    samples = list(samples)
    if rows_count < tacl * ucl:
        return None
    if rows_count + days >= 3000:
        seasons_n = nseas
    elif rows_count > 160:
        seasons_n = 1
        samples[0] = sum(samples[:nseas])
        nseas = 1
    else:
        return None
    if seasons_n > 1 and all(samples[s] < 100 * tacl for s in range(seasons_n)):
        seasons_n = 1
        samples[0] = sum(samples[:nseas])
        nseas = 1
    cont = [INVALID] * (len(samples) * tacl)
    for s in range(seasons_n):
        if samples[s] < 100 * tacl:
            continue
        ssi = sum(samples[:s])
        _qsort(rows, ssi, samples[s], 1)
        for c, (st, en) in enumerate(_windows(rows, ssi, samples[s], tacl, 1, False)):
            if st == -1 or en - st + 1 < 100:
                continue
            seg = rows[st:en + 1]
            xm = 0.0
            ym = 0.0
            for r in seg:
                xm += r[1]
                ym += r[2]
            xm /= len(seg)
            ym /= len(seg)
            sxy = sxx = syy = 0.0
            for r in seg:
                t1, t2 = r[1] - xm, r[2] - ym
                sxx = sxx + t1 * t1
                syy = syy + t2 * t2
                sxy = sxy + t1 * t2
            den = np.sqrt(sxx) * np.sqrt(syy)
            corr = sxy / den if den else float("nan")
            if corr != corr:
                corr = INVALID
            if abs(corr) > 0.5:
                continue
            _qsort(rows, st, en - st + 1, 2)
            um, fm = [], []
            for (a, b) in _windows(rows, st, en - st + 1, ucl, 2, True):
                su = sf = 0.0
                if a != -1:
                    for y in range(a, b + 1):
                        su += rows[y][2]
                        sf += rows[y][0]
                    su, sf = _div(su, b - a + 1), _div(sf, b - a + 1)
                um.append(su)
                fm.append(sf)
            if um[0] > 0.2:
                continue
            th = _fwd(um, fm, fn)
            cont[s * tacl + c] = th
            if seasons_n == 1:
                for i in range(nseas):
                    cont[i * tacl + c] = th
    return cont, nseas


def _year_frame(year=2023, freq='30min', seed=0, keep=None, sw_night=True):
    """One calendar year on its TIMESTAMP_MIDDLE grid with a u*-dependent night NEE.

    keep: optional bool mask of records that keep their NEE (the rest is NaN).
    """
    step = pd.Timedelta(freq)
    ix = pd.date_range(pd.Timestamp(year, 1, 1) + step / 2, pd.Timestamp(year + 1, 1, 1),
                       freq=freq, inclusive='left', name='TIMESTAMP_MIDDLE')
    n = len(ix)
    rng = np.random.RandomState(seed)
    hr = ix.hour + ix.minute / 60
    sw = np.clip(500 * np.sin(2 * np.pi * (hr - 6) / 24), 0, None) if sw_night else np.zeros(n)
    ustar = rng.uniform(0.02, 0.9, n)
    nee = np.where(ustar < 0.3, 1.0 + 10.0 * ustar, 4.0) + rng.normal(0, 0.3, n)
    if keep is not None:
        nee = np.where(keep, nee, np.nan)
    return pd.DataFrame({'NEE': nee, 'TA': rng.normal(10, 6, n), 'USTAR': ustar, 'SW_IN': sw},
                        index=ix)


def _det(df, **kw):
    return UstarMovingPointDetection(df, nee_col='NEE', ta_col='TA', ustar_col='USTAR',
                                     swin_col='SW_IN', verbose=0, **kw)


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
        self.assertEqual(d.get_annual_thresholds()["threshold"], float(np.nanmax(vals)))
        self.assertEqual(d.ta_class_thresholds_.shape, (7, 4))

    def test_kernels_match_literal_reference(self):
        """The numba kernels must reproduce the literal C transcription bitwise."""
        year = self.df.loc['2017']
        for taclasses, uclasses, fn in [(7, 20, 2), (7, 20, 1), (5, 15, 2), (7, 20, 3)]:
            d = UstarMovingPointDetection(year, ta_classes_count=taclasses,
                                          ustar_classes_count=uclasses, forward_mode_n=fn)
            p = d._prepare()
            rows = [list(r) for r in d._main_rows()]
            ref, nseas = _ustar_threshold(rows, p['rows_count'], p['days'], p['main_samples'],
                                          d.seasons_count, taclasses, uclasses, fn)
            got = d._main_run()
            self.assertEqual(got['sgc'], nseas)
            n = nseas * taclasses
            np.testing.assert_array_equal(
                got['container'][:n], np.array(ref[:n]),
                err_msg=f"mismatch for ta={taclasses}, ustar={uclasses}, fw={fn}")

    def test_main_run_order_is_season_month_record(self):
        """main.c:1014-1036: rows go season by season, month by month, then in record order."""
        d = UstarMovingPointDetection(self.df.loc['2017'],
                                      season_groups=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        p = d._prepare()
        order = p['main_order']
        month = d._month[order]
        expected_month_seq = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        rank = np.array([expected_month_seq.index(m) for m in month])
        self.assertTrue(np.all(np.diff(rank) >= 0))
        for m in np.unique(month):
            self.assertTrue(np.all(np.diff(order[month == m]) > 0))

    def test_qsort_tie_order_matches_ucrt(self):
        """Tie order of the C qsort, taken from the reference build (qsort_check)."""
        from diive.flux.lowres._ustar_mp_kernels import qsort_rows
        cases = [
            ([3, 1, 2, 1, 3, 1, 2, 2, 3, 1, 1, 2, 3, 3, 1, 2, 1, 3, 2, 1],
             [10, 1, 16, 3, 14, 5, 19, 9, 18, 7, 6, 11, 2, 15, 13, 12, 4, 17, 8, 0]),
            ([2, 2, 1, 1, 2, 1, 2, 1, 1], [7, 2, 3, 5, 8, 6, 4, 1, 0]),
        ]
        for keys, perm_c in cases:
            rows = np.column_stack([np.arange(len(keys)), np.array(keys, float), np.zeros(len(keys))])
            qsort_rows(rows, 0, len(keys), 1)
            self.assertEqual(rows[:, 0].astype(int).tolist(), perm_c)
            rows2 = [[i, float(k), 0.0] for i, k in enumerate(keys)]
            _qsort(rows2, 0, len(keys), 1)
            self.assertEqual([int(r[0]) for r in rows2], perm_c)

    def test_bootstrap_has_annual_row(self):
        d = UstarMovingPointDetection(self.df, verbose=0)
        bs = d.bootstrap(n_iter=15)
        self.assertIn("Annual", bs.index)
        self.assertEqual(len(bs), 5)  # 4 seasons + annual
        self.assertTrue(set(["mean", "std", "p05", "p50", "p95"]).issubset(bs.columns))

    def test_forward_mode_n_validation(self):
        with self.assertRaises(ValueError):
            UstarMovingPointDetection(self.df, forward_mode_n=0)

    def test_duplicate_months_are_refused(self):
        with self.assertRaises(ValueError):
            UstarMovingPointDetection(self.df, season_groups=[[1, 2], [2, 3]])


class TestOneFluxRecordRules(unittest.TestCase):
    """Record minimum and method choice of ustar.c:500-535 and main.c:1093."""

    def test_too_few_night_records_raise(self):
        # 130 night records: below ta_classes_count * ustar_classes_count = 140
        keep = np.zeros(17520, bool)
        keep[np.flatnonzero(_year_frame()['SW_IN'].to_numpy() < 10)[:130]] = True
        with self.assertRaises(ValueError) as ctx:
            _det(_year_frame(keep=keep)).detect()
        self.assertIn('140', str(ctx.exception))

    def test_short_record_with_few_nights_raises(self):
        # 150 night records and fewer than 3000 records in total
        df = _year_frame().iloc[:300]
        det = _det(df)
        p = det._prepare()
        self.assertTrue(140 <= p['rows_count'] <= 160, p['rows_count'])
        with self.assertRaises(ValueError) as ctx:
            det.detect()
        self.assertIn('Insufficient', str(ctx.exception))

    def test_short_record_with_many_nights_is_one_big_season(self):
        # all records night, 2000 of them: night + day < 3000 but > 160 night records
        df = _year_frame(sw_night=False).iloc[:2000]
        det = _det(df)
        res = det.detect()
        self.assertTrue(det.one_big_season_)
        self.assertEqual(res['threshold'].nunique(dropna=False), 1)
        self.assertTrue(np.isfinite(res['threshold'].iloc[0]))

    def test_all_seasons_small_is_one_big_season(self):
        # 3000+ records, but every season below 700 night records
        rng = np.random.RandomState(1)
        night = np.flatnonzero(_year_frame()['SW_IN'].to_numpy() < 10)
        keep = np.zeros(17520, bool)
        keep[rng.choice(night, 2000, replace=False)] = True
        det = _det(_year_frame(keep=keep))
        det.detect()
        self.assertTrue(det.one_big_season_)

    def test_seasonal_detection_skips_small_seasons(self):
        # Q3 alone above 700: seasonal, the other seasons get no threshold
        df = _year_frame()
        month = df.index.month.to_numpy()
        keep = np.isin(month, [7, 8, 9])
        det = _det(_year_frame(keep=keep))
        res = det.detect()['threshold']
        self.assertFalse(det.one_big_season_)
        self.assertTrue(np.isfinite(res.iloc[2]))
        self.assertTrue(res.drop(res.index[2]).isna().all())


class TestOneFluxBootstrap(unittest.TestCase):

    def test_bootstrap_seasons_by_row_position_with_leap_quirk(self):
        """main.c:1018-1019: a leap February adds 1 row, not 48."""
        det = _det(_year_frame(year=2024))
        p = det._prepare()
        self.assertTrue(p['year_grid'])
        s = p['boot_season']
        # Q1 ends after (31 + 28 + 31) * 48 + 1 = 4321 rows, 47 rows before March ends
        self.assertEqual(s[4320], 0)
        self.assertEqual(s[4321], 1)
        # the last 47 rows lie beyond the last season end and count to season 0
        self.assertTrue(np.all(s[-47:] == 0))
        self.assertEqual(s[-48], 3)
        # non-leap year: boundaries on the month ends
        s = _det(_year_frame(year=2023))._prepare()['boot_season']
        self.assertEqual((s[4319], s[4320]), (0, 1))
        self.assertTrue(np.all(s[-48:] == 3))

    def test_end_and_middle_stamps_give_the_same_result(self):
        df = _year_frame(year=2024)
        end = df.copy()
        end.index = end.index + pd.Timedelta('15min')
        a, b = _det(df), _det(end)
        self.assertEqual(a.detect()['threshold'].tolist(), b.detect()['threshold'].tolist())
        np.testing.assert_array_equal(a.bootstrap_annual_samples(5), b.bootstrap_annual_samples(5))

    def test_identity_draw_reproduces_the_main_run(self):
        """Drawing every row once gives the main-run threshold (non-leap year)."""
        df = _year_frame(year=2023)
        det = _det(df)
        det.detect()
        idx = np.arange(len(df))[None, :]
        val = det.bootstrap_annual_samples(indices=idx)
        self.assertEqual(val.shape, (1,))
        self.assertEqual(val[0], det.get_annual_thresholds()['threshold'])

    def test_indices_hook_and_seed_are_deterministic(self):
        det = _det(_year_frame())
        rng_draws = det.bootstrap_annual_samples(6, rng=np.random.default_rng(3))
        again = det.bootstrap_annual_samples(6, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(rng_draws, again)
        idx = np.random.default_rng(3).integers(0, 17520, (6, 17520))
        np.testing.assert_array_equal(det.bootstrap_annual_samples(indices=idx),
                                      det.bootstrap_annual_samples(indices=idx))
        # default generator is seeded with random_state
        np.testing.assert_array_equal(det.bootstrap_annual_samples(4), det.bootstrap_annual_samples(4))
        with self.assertRaises(ValueError):
            det.bootstrap_annual_samples(indices=np.array([[0, 17520]]))

    def test_failed_draw_is_nan_and_flags_the_drop(self):
        """bootstrapping.c:194-201: a failed draw makes ONEFlux drop the whole bootstrap."""
        night = np.flatnonzero(_year_frame()['SW_IN'].to_numpy() < 10)
        keep = np.zeros(17520, bool)
        keep[night[:150]] = True
        det = _det(_year_frame(keep=keep))
        p = det._prepare()
        good = np.arange(17520)
        # a draw with only 100 of the night records: fewer than 140
        bad = np.where(np.isin(good, night[100:150]), night[200], good)
        vals = det.bootstrap_annual_samples(indices=np.vstack([good, bad, good]))
        self.assertEqual(len(vals), 3)
        self.assertEqual(det.bootstrap_failed_.tolist(), [False, True, False])
        self.assertTrue(det.bootstrap_dropped_)
        self.assertTrue(np.isnan(vals[1]))
        self.assertEqual(p['rows_count'], 150)

    def test_collapse_to_one_season_is_sticky(self):
        """ustar.c:509-533 change the global season count, so later draws stay collapsed."""
        df = _year_frame()
        month = df.index.month.to_numpy()
        night = df['SW_IN'].to_numpy() < 10
        q3 = np.flatnonzero(night & np.isin(month, [7, 8, 9]))
        q1 = np.flatnonzero(night & np.isin(month, [1, 2, 3]))
        keep = np.zeros(len(df), bool)
        keep[q3[:700]] = True
        keep[q1[:200]] = True
        det = _det(_year_frame(keep=keep))
        identity = np.arange(len(df))
        # one Q3 night record replaced by a day record: Q3 drops to 699, all seasons < 700
        small = identity.copy()
        small[q3[0]] = np.flatnonzero(~night)[0]
        season_med, _ = det._run_bootstrap(None, None, np.vstack([identity, small, identity]))
        self.assertTrue(np.isfinite(season_med[0, 2]) and np.isnan(season_med[0, 0]))
        # after the collapse the identity draw is pooled into one big season
        self.assertTrue(np.all(season_med[2] == season_med[2, 0]))
        self.assertTrue(np.isfinite(season_med[2, 0]))
        self.assertNotEqual(season_med[2, 0], season_med[0, 2])


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
        # 250 records, about 130 at night: ONEFlux skips such a year (main.c:1093).
        # This used to be MIN_SAMPLES_PERIOD // 3 = 1000 records, but ONEFlux does not
        # refuse 1000 records with about 500 nights: it pools them into one big season.
        n = 250
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
            # 120 records per year (was 800): with about 800 nights in total ONEFlux
            # pools one big season instead of refusing the record
            ix = pd.date_range(f'{y}-01-01', periods=120, freq='30min',
                               name='TIMESTAMP_MIDDLE')
            rng = np.random.RandomState(y)
            hr = ix.hour + ix.minute / 60
            sw = np.clip(500 * np.sin(2 * np.pi * (hr - 6) / 24), 0, None)
            frames.append(pd.DataFrame(
                {'NEE': rng.randn(120), 'TA': 10 + 5 * rng.randn(120),
                 'USTAR': np.abs(rng.randn(120)) * 0.5, 'SW_IN': sw}, index=ix))
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

    def test_threshold_zero_flags_nothing_not_even_missing_ustar(self):
        # thresholds=[0] with CUT_NONE is the no-filter setting for H and LE (L347)
        from diive.flux.lowres.ustarthreshold import FlagSingleConstantUstarThreshold
        nee, ustar = self._frame()
        flag = FlagSingleConstantUstarThreshold(series=nee, ustar=ustar, threshold=0).run().get_flag()
        self.assertEqual(flag.iloc[0], 0)   # flux present, USTAR unknown
        self.assertEqual(flag.iloc[4], 0)
        self.assertEqual(flag.iloc[2], 0)
        self.assertEqual(int((flag == 2).sum()), 0)
        self.assertTrue(np.isnan(flag.iloc[1]))   # missing flux stays NaN

    def test_threshold_series_with_zeros_keeps_those_records_only(self):
        import pandas as pd
        from diive.flux.lowres.ustarthreshold import FlagMultipleVariableUstarThresholds
        nee, ustar = self._frame()
        thr = pd.Series([0.0, 0.3, 0.3, 0.3, 0.3, 0.3], index=nee.index)
        flagger = FlagMultipleVariableUstarThresholds(
            series=nee, ustar=ustar, threshold_series={'VUT_50': thr},
            showplot=False, verbose=False)
        flagger.calc()
        flagcol = [c for c in flagger.results.columns if c.startswith('FLAG_')][0]
        flag = flagger.results[flagcol]
        self.assertEqual(flag.iloc[0], 0)   # threshold 0: kept without USTAR
        self.assertEqual(flag.iloc[4], 2)   # threshold 0.3: rejected without USTAR
        self.assertEqual(flag.iloc[2], 2)   # measured USTAR below 0.3


class TestScenarioPlotCountsAreAddressedPositionally(unittest.TestCase):
    """The bar annotations index a label-indexed count Series.

    `describe().loc['count']` is indexed by column name, so a positional lookup
    on it raises KeyError on pandas 3. Plotting is this class's only purpose.
    """

    def test_the_scenario_plot_annotates_every_bar(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        from diive.flux.lowres.ustarthreshold import UstarThresholdConstantScenarios
        ix = pd.date_range('2023-06-01 00:30', periods=480, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(0)
        nee = pd.Series(rng.normal(0, 3, len(ix)), index=ix, name='NEE')
        ustar = pd.Series(rng.uniform(0, 0.8, len(ix)), index=ix, name='USTAR')
        swinpot = pd.Series(np.where((ix.hour >= 8) & (ix.hour <= 18), 500.0, 0.0),
                            index=ix, name='SW_IN_POT')
        scenarios = UstarThresholdConstantScenarios(series=nee, ustar=ustar, swinpot=swinpot)
        try:
            scenarios.calc(ustarthresholds=[0.1, 0.2], showplot=True)
            texts = [t.get_text() for t in plt.gcf().axes[0].texts]
        finally:
            plt.close('all')
        # One count + one percentage per bar: unfiltered series + two thresholds.
        self.assertEqual(len(texts), 6)
        # The unfiltered column is the reference, so its bar reads 100%.
        self.assertIn('100%', texts)


class TestConstantScenariosRejectMissingUstar(unittest.TestCase):
    """A record with no USTAR cannot pass a u* threshold, as in the flag and ONEFlux."""

    def test_missing_ustar_is_removed_in_every_scenario(self):
        import pandas as pd
        from diive.flux.lowres.ustarthreshold import UstarThresholdConstantScenarios
        ix = pd.date_range('2023-06-01 00:30', periods=6, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        nee = pd.Series(np.arange(1.0, 7.0), index=ix, name='NEE')
        ustar = pd.Series([0.05, 0.25, np.nan, 0.45, np.nan, 0.6], index=ix, name='USTAR')
        swinpot = pd.Series(0.0, index=ix, name='SW_IN_POT')
        scenarios = UstarThresholdConstantScenarios(series=nee, ustar=ustar, swinpot=swinpot)
        scenarios.calc(ustarthresholds=[0.1, 0.5])
        df = scenarios.scenariosdf
        self.assertEqual(df['NEE_CUT0.1'].dropna().tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(df['NEE_CUT0.5'].dropna().tolist(), [6.0])
        # The unfiltered column keeps every record.
        self.assertEqual(df['NEE'].notna().sum(), 6)


class TestBootstrapThresholdsAreReproducible(unittest.TestCase):
    """L97: UstarBootstrapThresholds resampled unseeded, in both of its paths.

    The generic loop called `df_window.sample()` with no random_state, and the fast path
    never passed the `rng` that `bootstrap_annual_samples` already accepted. So VUT and
    CUT thresholds moved between runs — and this is the class `run_chain` uses for CUT
    detection, unlike the standalone Vekuri detector fixed in L86.

    The seed is derived per window year rather than shared, which is what lets the serial
    and parallel paths agree: a single shared seed would make every window resample the
    same positions, correlating years that are meant to be independent draws.
    """

    @staticmethod
    def _df(years=(2021, 2022, 2023), per_year=2880, seed=0):
        import pandas as pd
        rng = np.random.RandomState(seed)
        frames = []
        for y in years:
            ix = pd.date_range(f'{y}-06-01 00:15', periods=per_year, freq='30min',
                               name='TIMESTAMP_MIDDLE')
            ustar = rng.uniform(0, 0.9, per_year)
            nee = np.where(ustar < 0.25, -1.0 - 4.0 * ustar, -2.0) + rng.normal(0, 0.3, per_year)
            frames.append(pd.DataFrame({'NEE': nee, 'TA': rng.normal(10, 6, per_year),
                                        'USTAR': ustar, 'SW_IN': 0.0}, index=ix))
        return pd.concat(frames)

    def _run(self, **kwargs):
        from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
        from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
        boot = UstarBootstrapThresholds(
            self._df(), detector_class=UstarVekuriThresholdDetection,
            detector_kwargs=dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR',
                                 swin_col='SW_IN'),
            n_iter=3, verbose=0, **kwargs)
        annual = boot.run()
        return annual, boot.get_cut_threshold()

    def test_two_runs_with_the_same_seed_agree(self):
        import pandas as pd
        (a_vut, a_cut), (b_vut, b_cut) = self._run(), self._run()
        pd.testing.assert_frame_equal(a_vut, b_vut)
        self.assertEqual(a_cut, b_cut)

    def test_a_different_seed_gives_a_different_draw(self):
        """Guards the other direction: the seed must actually reach both paths."""
        a_vut, _ = self._run(random_state=42)
        b_vut, _ = self._run(random_state=7)
        self.assertFalse(a_vut.equals(b_vut))

    def test_serial_and_parallel_agree(self):
        """The point of deriving the seed per year: n_jobs must not change the answer.

        With one shared seed this would still be deterministic, but every window would
        draw the same positions. With no seed the two paths would simply disagree.
        """
        import pandas as pd
        serial_vut, serial_cut = self._run(n_jobs=1)
        parallel_vut, parallel_cut = self._run(n_jobs=2)
        pd.testing.assert_frame_equal(serial_vut, parallel_vut)
        self.assertEqual(serial_cut, parallel_cut)

    def test_the_seed_is_derived_per_year(self):
        """Two windows must not draw the same positions, or the years are not independent."""
        from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
        from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
        boot = UstarBootstrapThresholds(
            self._df(), detector_class=UstarVekuriThresholdDetection,
            detector_kwargs=dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR',
                                 swin_col='SW_IN'), n_iter=2, random_state=42, verbose=0)
        seeds = [boot._seed_for(y) for y in boot.years_]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertIsNone(UstarBootstrapThresholds(
            self._df(), detector_class=UstarVekuriThresholdDetection,
            detector_kwargs=dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR',
                                 swin_col='SW_IN'), random_state=None,
            verbose=0)._seed_for(2021))


class TestVekuriBootstrapIsReproducible(unittest.TestCase):
    """L86: bootstrap() resampled with no random_state, so its percentiles moved.

    These thresholds feed u* filtering and therefore the whole flux chain, while the
    rf/xgb seeds are pinned precisely because output drifts without them. `random_state`
    (default 42) seeds the draws, offset per iteration so they still differ from each
    other; pass None for the old non-deterministic behaviour.
    """

    @staticmethod
    def _df(seed=0, n=2880):
        import pandas as pd
        ix = pd.date_range('2023-01-01 00:30', periods=n, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(seed)
        ustar = rng.uniform(0, 0.9, n)
        # A real u*-dependent NEE, so a threshold is actually findable.
        nee = np.where(ustar < 0.25, -1.0 - 4.0 * ustar, -2.0) + rng.normal(0, 0.3, n)
        return pd.DataFrame({'NEE': nee, 'TA': rng.normal(10, 6, n),
                             'USTAR': ustar, 'SW_IN': 0.0}, index=ix)

    def _run(self, **kwargs):
        from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
        det = UstarVekuriThresholdDetection(df=self._df(), nee_col='NEE', ta_col='TA',
                                            ustar_col='USTAR', swin_col='SW_IN',
                                            verbose=0, **kwargs)
        det.detect()
        return det.bootstrap(n_iter=6)

    def test_two_runs_with_the_same_seed_agree(self):
        import pandas as pd
        a, b = self._run(), self._run()
        pd.testing.assert_frame_equal(a, b)

    def test_a_different_seed_gives_a_different_draw(self):
        """Guards the other direction: the seed must actually reach `sample`."""
        a = self._run(random_state=42)
        b = self._run(random_state=7)
        self.assertFalse(a.equals(b))


class TestVekuriSummaryBeforeDetect(unittest.TestCase):
    """summary() reports that detection has not run; it must not fail in that guard.

    `results_` starts as an empty DataFrame (as in UstarMovingPointDetection), so
    `.empty` answers the question instead of raising AttributeError on a dict.
    """

    def test_summary_says_detect_first(self):
        import pandas as pd
        from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
        ix = pd.date_range('2023-01-01 00:30', periods=480, freq='30min',
                           name='TIMESTAMP_MIDDLE')
        rng = np.random.RandomState(0)
        df = pd.DataFrame({'NEE': rng.normal(0, 3, len(ix)), 'TA': rng.normal(10, 5, len(ix)),
                           'USTAR': rng.uniform(0, 0.8, len(ix)), 'SW_IN': 0.0}, index=ix)
        det = UstarVekuriThresholdDetection(df=df, nee_col='NEE', ta_col='TA',
                                            ustar_col='USTAR', swin_col='SW_IN', verbose=0)
        self.assertIn('detect()', det.summary())
        # Documented attribute, so it must exist before bootstrap() too.
        self.assertTrue(det.bootstrap_stats_.empty)
