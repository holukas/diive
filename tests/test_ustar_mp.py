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
