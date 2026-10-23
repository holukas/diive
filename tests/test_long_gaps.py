"""Tests for the ONEFlux long-gap removal (finding L397).

The mask is checked bitwise against ONEFlux's own ``get_indices_to_filter``
(``oneflux/pipeline/site_data_product.py``), copied below without its logging.
"""
import unittest

import numpy
import numpy as np
import pandas as pd

from diive.gapfilling.longgaps import oneflux_long_gap_mask, oneflux_long_gap_periods


def get_indices_to_filter(qcdata, qc_threshold=2, window_size=15*48, minimum_gap=5*48):
    '''
    Fast creation of mask for QC flags above threshold value
    continually within window size
    '''
    # ONEFlux site_data_product.py, verbatim except for the log calls.
    # filter QC array, minimum quality acceptable
    qc_filtered = (qcdata > qc_threshold).astype(int)

    # prepare window array of 1s for convolution
    conv_window = [1,] * window_size

    # create array counting how many of entries match condition
    # (low quality, i.e., qc values higher than threshold)
    conv = numpy.convolve(qc_filtered, conv_window, mode='valid')

    # create array of start indices where number of entries match window size
    indices_start = numpy.where(conv == window_size)[0]

    # create (start, end) pairs for windows
    indices = [numpy.arange(index, index + window_size) for index in indices_start]

    # flatten all entries (and remove potential duplicates)
    indices_unique = numpy.unique(indices)

    # merge all contiguous windows into longer tuples
    # use (i[0]:i[-1] + 1) to access indexes for each window of interest
    indices_tmp = numpy.where(numpy.diff(indices_unique) != 1)[0] + 1
    indices_contiguous = numpy.split(indices_unique, indices_tmp)
    # remove empty result arrays
    # TODO: revise logic to avoid empty arrays in the first place
    indices_contiguous_clean = []
    for a in indices_contiguous:
        if a.size > 0:
            indices_contiguous_clean.append(a)
    indices_contiguous = indices_contiguous_clean

    # check length of pairs against window size minimums
    # (i.e., start at record window_size+1) and taking into
    # account the minimum gap size
    indices_clean = []
    for index in indices_contiguous:
        (first, last) = (index[0], index[-1])
        if first == 0:
            # at the start of the record
            if last == window_size:
                pass
            elif last > window_size:
                indices_clean.append((first, last - window_size + 1))
        elif last == qcdata.size - 1:
            # at the end of the record
            if first == qcdata.size - 1 - window_size:
                pass
            elif first < qcdata.size - 1 - window_size:
                indices_clean.append((first + window_size - 1, last))
        else:
            # in the middle of the record, also check for minimum gap size
            if (last - first + 1) <= ((2 * window_size) + minimum_gap):
                pass
            else:
                indices_clean.append((first + window_size, last - window_size))

    # create mask to be used to set values to NaN in True positions
    mask = numpy.isnan(qcdata)
    mask[:] = False
    for (first, last) in indices_clean:
        mask[first:last + 1] = True
    return mask


def _qc_with_runs(n, runs, rng=None):
    """QC array of n records, 0/1/2 outside the runs, 3 in the runs (start, length)."""
    qc = (rng.integers(0, 3, n) if rng is not None else np.zeros(n)).astype(float)
    for start, length in runs:
        qc[max(start, 0):max(start + length, 0)] = 3.0
    return qc


class TestLongGapMaskAgainstOneflux(unittest.TestCase):

    def _check(self, qc, rpd, msg=''):
        w, g = 15 * rpd, 5 * rpd
        expected = get_indices_to_filter(qc.copy(), qc_threshold=2, window_size=w,
                                         minimum_gap=g)
        got = oneflux_long_gap_mask(qc, records_per_day=rpd)
        np.testing.assert_array_equal(got, expected, err_msg=msg)
        return got

    def test_hand_made_cases(self):
        for rpd in (48, 24):
            w, g = 15 * rpd, 5 * rpd
            n = 120 * rpd
            mid = 40 * rpd
            cases = {
                # at the start: lengths around the window and window + 1
                **{f'start_{d}': [(0, w + d)] for d in (-1, 0, 1, 2, 3, 100)},
                # from the second record (not the start)
                **{f'second_{d}': [(1, w + d)] for d in (-1, 0, 1, 2, 3, 100)},
                # at the end
                **{f'end_{d}': [(n - w - d, w + d)] for d in (-1, 0, 1, 2, 3, 100)},
                # up to the second-to-last record (not the end)
                **{f'penultimate_{d}': [(n - 1 - w - d, w + d)] for d in (-1, 0, 1, 2, 100)},
                # in the middle, around 2 w + g
                **{f'middle_{d}': [(mid, 2 * w + g + d)] for d in (-2, -1, 0, 1, 2, 50)},
                'middle_window': [(mid, w)],
                'several': [(0, w + 5), (30 * rpd, 2 * w + g + 7), (80 * rpd, w + 1),
                            (n - w - 9, w + 9)],
                'adjacent': [(mid, 2 * w + g + 1), (mid + 2 * w + g + 2, 2 * w + g + 1)],
                'whole_record': [(0, n)],
                'none': [],
            }
            for name, runs in cases.items():
                with self.subTest(rpd=rpd, case=name):
                    self._check(_qc_with_runs(n, runs), rpd, name)
        # A run of w + 2 records loses 3 at the edges of the record, w - 1
        # stay next to the data. In the middle w stay at each end.
        rpd, w, g = 48, 720, 240
        m = oneflux_long_gap_mask(_qc_with_runs(4000, [(0, w + 2)]), records_per_day=rpd)
        self.assertEqual(np.flatnonzero(m).tolist(), [0, 1, 2])
        m = oneflux_long_gap_mask(_qc_with_runs(4000, [(100, 2 * w + g + 2)]),
                                  records_per_day=rpd)
        self.assertEqual(np.flatnonzero(m).tolist(), [100 + w + i for i in range(g + 2)])
        m = oneflux_long_gap_mask(_qc_with_runs(4000, [(4000 - w - 2, w + 2)]),
                                  records_per_day=rpd)
        self.assertEqual(np.flatnonzero(m).tolist(), [3997, 3998, 3999])

    def test_record_shorter_than_the_window(self):
        for n in (1, 5, 719, 720, 721, 722):
            with self.subTest(n=n):
                self._check(np.full(n, 3.0), 48)

    def test_random_arrays(self):
        rng = np.random.default_rng(42)
        for i in range(300):
            rpd = (48, 24)[i % 2]
            n = int(rng.integers(rpd, 400 * rpd))
            runs = [(int(rng.integers(-5 * rpd, n)), int(rng.integers(1, 60 * rpd)))
                    for _ in range(int(rng.integers(0, 8)))]
            qc = _qc_with_runs(n, runs, rng)
            # missing QC and other values above the threshold
            qc[rng.random(n) < 0.002] = np.nan
            qc[rng.random(n) < 0.001] = -9999.9
            with self.subTest(i=i):
                self._check(qc, rpd, f'case {i}')

    def test_series_input_and_options(self):
        idx = pd.date_range('2021-01-01 00:15', periods=60 * 48, freq='30min')
        qc = pd.Series(_qc_with_runs(len(idx), [(0, 20 * 48)]), index=idx, name='q')
        m = oneflux_long_gap_mask(qc)
        self.assertIsInstance(m, pd.Series)
        self.assertTrue(m.index.equals(idx))
        np.testing.assert_array_equal(m.to_numpy(), self._check(qc.to_numpy(), 48))
        hourly = qc.resample('60min').max()
        np.testing.assert_array_equal(oneflux_long_gap_mask(hourly).to_numpy(),
                                      self._check(hourly.to_numpy(), 24))
        # other settings
        np.testing.assert_array_equal(
            oneflux_long_gap_mask(qc, qc_threshold=1, window_days=10, minimum_gap_days=2),
            get_indices_to_filter(qc.to_numpy(), 1, 480, 96))
        with self.assertRaisesRegex(ValueError, 'records_per_day'):
            oneflux_long_gap_mask(qc.to_numpy())


class TestLongGapPeriods(unittest.TestCase):

    def test_days_weeks_months(self):
        starts = pd.DatetimeIndex(['2021-01-07 23:30', '2021-03-01 00:00'])
        days = pd.date_range('2021-01-01', '2021-12-31', freq='D')
        dd = oneflux_long_gap_periods(starts, days, days)
        self.assertEqual(days[dd].strftime('%Y%m%d').tolist(), ['20210107', '20210301'])
        wk_first = pd.DatetimeIndex(['2021-01-01', '2021-01-08', '2021-02-26'])
        wk_last = pd.DatetimeIndex(['2021-01-07', '2021-01-14', '2021-03-04'])
        self.assertEqual(oneflux_long_gap_periods(starts, wk_first, wk_last).tolist(),
                         [True, False, True])
        self.assertEqual(oneflux_long_gap_periods(starts[:0], wk_first, wk_last).tolist(),
                         [False, False, False])


def _mds_case():
    """90 days of half-hourly data with a 50-day gap in flux and drivers: MDS
    fills it from the diurnal cycle, with quality 3 more than 5 days from data."""
    n = 90 * 48
    idx = pd.date_range('2021-03-01 00:15', periods=n, freq='30min')
    rng = np.random.RandomState(1)
    hr = idx.hour + idx.minute / 60
    sw = np.clip(800 * np.sin((hr - 6) / 12 * np.pi), 0, None) + rng.uniform(0, 50, n)
    ta = 10 + 5 * np.sin((hr - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, n)
    vpd = np.clip(0.8 + 0.4 * np.sin((hr - 9) / 24 * 2 * np.pi), 0, None)
    flux = -0.02 * sw + 0.3 * ta + rng.normal(0, 2, n)
    df = pd.DataFrame({'flux': flux, 'sw': sw, 'ta': ta, 'vpd': vpd}, index=idx)
    df.loc[rng.uniform(size=n) < 0.2, 'flux'] = np.nan
    df.iloc[20 * 48:70 * 48] = np.nan
    return df


class TestFluxMDSLongGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from diive.gapfilling.mds import FluxMDS
        cls.df = _mds_case()
        cls.runs = {}
        for key, kw in {'oneflux': dict(mode='oneflux'),
                        'oneflux_fill': dict(mode='oneflux', fill_long_gaps=True),
                        'diive': dict(mode='diive'),
                        'diive_cut': dict(mode='diive', fill_long_gaps=False),
                        'reddyproc': dict(mode='reddyproc')}.items():
            m = FluxMDS(df=cls.df, flux='flux', swin='sw', ta='ta', vpd='vpd', verbose=0, **kw)
            m.run()
            cls.runs[key] = m

    def test_default_per_mode(self):
        self.assertFalse(self.runs['oneflux'].fill_long_gaps)
        self.assertTrue(self.runs['oneflux_fill'].fill_long_gaps)
        self.assertTrue(self.runs['diive'].fill_long_gaps)
        self.assertFalse(self.runs['diive_cut'].fill_long_gaps)
        self.assertTrue(self.runs['reddyproc'].fill_long_gaps)

    def test_mask_is_the_oneflux_rule_on_the_mds_qc(self):
        m = self.runs['oneflux']
        qc = m.get_qc(full=True)
        gdf = m.gapfilling_df_
        # QC as nee_proc writes it: 0 measured, else the MDS quality
        measured = gdf['flux'].notna().to_numpy()
        np.testing.assert_array_equal(qc.to_numpy()[measured], 0.0)
        np.testing.assert_array_equal(qc.to_numpy()[~measured],
                                      gdf['.PREDICTIONS_QUALITY'].to_numpy()[~measured])
        expected = get_indices_to_filter(qc.to_numpy(), 2, 720, 240)
        np.testing.assert_array_equal(m.long_gap_mask_.to_numpy(), expected)
        # 50-day gap: quality 3 more than 5 days from the data, of which the
        # inner part beyond 15 days at each end is removed
        n_removed = int(m.long_gap_mask_.sum())
        self.assertGreater(n_removed, 0)
        self.assertLess(n_removed, 20 * 48)
        self.assertFalse(m.long_gap_mask_.iloc[:20 * 48].any())
        self.assertFalse(m.long_gap_mask_.iloc[70 * 48:].any())

    def test_fill_true_keeps_everything(self):
        full, cut = self.runs['oneflux_fill'], self.runs['oneflux']
        # Same cascade: all columns except the gap-filled series and its flag agree
        same = [c for c in full.gapfilling_df_.columns
                if c not in (full.target_gapfilled, full.target_gapfilled_flag)]
        pd.testing.assert_frame_equal(full.gapfilling_df_[same], cut.gapfilling_df_[same])
        self.assertEqual(full.scores_, cut.scores_)
        self.assertEqual(int(full.get_gapfilled_target().isna().sum()), 0)
        pd.testing.assert_series_equal(full.get_gapfilled_target(),
                                       full.get_gapfilled_target(full=True))
        pd.testing.assert_series_equal(full.long_gap_mask_, cut.long_gap_mask_)

    def test_fill_false_sets_nan_only_inside_the_mask(self):
        full, cut = self.runs['oneflux_fill'], self.runs['oneflux']
        mask = cut.long_gap_mask_.to_numpy()
        for get in ('get_gapfilled_target', 'get_flag', 'get_qc'):
            with self.subTest(get=get):
                a = getattr(full, get)()
                b = getattr(cut, get)()
                self.assertTrue(b[mask].isna().all())
                self.assertTrue(a[mask].notna().all())
                pd.testing.assert_series_equal(a[~mask], b[~mask])
                pd.testing.assert_series_equal(getattr(cut, get)(full=True), a)
        # Same in the result frame
        self.assertTrue(cut.gapfilling_df_[cut.target_gapfilled][mask].isna().all())
        # The mode decides only the default: the diive mode masks the same way
        d, dc = self.runs['diive'], self.runs['diive_cut']
        pd.testing.assert_series_equal(d.get_gapfilled_target(full=True),
                                       dc.get_gapfilled_target(full=True))
        dmask = dc.long_gap_mask_.to_numpy()
        self.assertTrue(dmask.any())
        self.assertTrue(dc.get_gapfilled_target()[dmask].isna().all())
        self.assertEqual(int(d.get_gapfilled_target().isna().sum()), 0)


class TestNeePercentileQc(unittest.TestCase):

    def test_by_hand(self):
        from diive.flux.lowres.ustar_reference import nee_percentile_qc
        idx = pd.date_range('2021-01-01 00:15', periods=3, freq='30min')
        # row 0: values 39..0, QC = value mod 4; row 1: ties; row 2: none valid
        v = np.vstack([np.arange(40.0)[::-1], np.zeros(40), np.full(40, np.nan)])
        q = np.vstack([np.arange(40.0)[::-1] % 4, np.arange(40) % 4, np.zeros(40)])
        v[0, 5] = np.nan  # value 34 missing: 39 valid
        out = nee_percentile_qc(pd.DataFrame(v, index=idx), pd.DataFrame(q, index=idx))
        self.assertEqual(list(out.columns), ['05', '16', '25', '50', '75', '84', '95'])
        # 39 valid sorted values 0..33, 35..39: position ROUND(p/100*39) - 1
        srt = [x for x in range(40) if x != 34]
        for p in (5, 16, 25, 50, 75, 84, 95):
            pos = max(int(p / 100.0 * 39 + 0.5) - 1, 0)
            self.assertEqual(out.loc[idx[0], f'{p:02d}'], srt[pos] % 4)
        # Ties keep the version order (stable sort)
        for p in (5, 16, 25, 50, 75, 84, 95):
            pos = max(int(p / 100.0 * 40 + 0.5) - 1, 0)
            self.assertEqual(out.loc[idx[1], f'{p:02d}'], pos % 4)
        self.assertTrue(out.loc[idx[2]].isna().all())

    def test_period_days(self):
        from diive.flux.lowres.ustar_reference import oneflux_period_days
        p = oneflux_period_days([2020, 2021])
        self.assertEqual(len(p['DD'][0]), 366 + 365)
        first, last = p['WW']
        self.assertEqual(len(first), 104)
        self.assertEqual((first[0], last[0]),
                         (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-07')))
        self.assertEqual((first[51], last[51]),
                         (pd.Timestamp('2020-12-23'), pd.Timestamp('2020-12-31')))
        self.assertEqual((first[103], last[103]),
                         (pd.Timestamp('2021-12-24'), pd.Timestamp('2021-12-31')))
        self.assertEqual(p['MM'][1][1], pd.Timestamp('2020-02-29'))
        self.assertEqual(list(p['YY'][0]),
                         [pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])


def _chain_with_long_gap():
    """Example month to Level 3.3 with the 40 REF percentiles. Flux and MDS
    drivers are missing in the first 20 days, so MDS fills them with quality 3
    and removes the start of the month as long gap. Partitioning gets its own
    complete drivers."""
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    from diive.flux.lowres.ustar_reference import ONEFLUX_REF_PERCENTILES
    from diive.flux.postprocessingchain import (
        init_flux_data, make_level32_detector, run_level2, run_level31, run_level32,
        run_level33_ustar_detection,
    )
    df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
    for c in ('TA_1_1_1', 'SW_IN_1_1_1'):
        df[c] = df[c].bfill()
    df['VPD_kPa'] = df['VPD_EP'].bfill() * 0.1
    df['TA_PART'] = df['TA_1_1_1']
    df['SW_PART'] = df['SW_IN_1_1_1']
    df.iloc[:20 * 48, [df.columns.get_loc(c) for c in
                       ('FC', 'TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa')]] = np.nan
    data = init_flux_data(df=df, fluxcol='FC', site_lat=46.58, site_lon=9.79, utc_offset=1)
    data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
    data = run_level31(data)
    data, sod = make_level32_detector(data)
    sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
    sod.addflag()
    data = run_level32(data, outlier_detector=sod)
    thr = {16: 0.10, 50: 0.15, 84: 0.20}
    thr.update({p: 0.05 + 0.002 * p for p in ONEFLUX_REF_PERCENTILES})
    return run_level33_ustar_detection(data, thresholds=thr, reference=True,
                                       showplot=False, verbose=False)


_MDS = dict(swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa', mode='oneflux')
_PART = dict(method='nighttime', ta='TA_PART', sw_in='SW_PART', ta_f='TA_PART', verbose=0)


class TestChainLongGaps(unittest.TestCase):
    """REF selection and partitioning use the full fill, as ONEFlux does, and
    only the output columns lose the long gaps."""

    @classmethod
    def setUpClass(cls):
        from diive.flux.postprocessingchain import (
            run_level41_mds, run_level41_ustar_reference, run_level42_partitioning,
            run_level42_ustar_reference,
        )
        l33 = _chain_with_long_gap()
        cls.out = {}
        for key, fill in (('full', True), ('cut', None)):
            d = run_level41_mds(l33, fill_long_gaps=fill, **_MDS)
            d = run_level41_ustar_reference(d, verbose=False)
            d = run_level42_partitioning(d, **_PART)
            d = run_level42_ustar_reference(d, **_PART)
            cls.out[key] = d

    def _mask(self, scen):
        return self.out['cut'].levels.level41_mds[scen].long_gap_mask_.to_numpy()

    def _check_column(self, col, mask):
        a, b = self.out['full'].fpc_df[col], self.out['cut'].fpc_df[col]
        self.assertTrue(b[mask].isna().all(), col)
        pd.testing.assert_series_equal(a[~mask], b[~mask])

    def test_mds_columns(self):
        full, cut = self.out['full'], self.out['cut']
        self.assertTrue(all(i.fill_long_gaps for i in full.levels.level41_mds.values()))
        self.assertFalse(any(i.fill_long_gaps for i in cut.levels.level41_mds.values()))
        self.assertGreater(int(self._mask('CUT_50').sum()), 0)
        for scen, gf_col in cut.gapfilled_cols()['mds'].items():
            inst = cut.levels.level41_mds[scen]
            with self.subTest(scen=scen):
                mask = self._mask(scen)
                self._check_column(gf_col, mask)
                self._check_column(inst.target_gapfilled_flag, mask)
                self.assertTrue(full.fpc_df[gf_col][mask].notna().all())

    def test_nee_ref_selected_on_the_full_fill(self):
        a = self.out['full'].levels.ustar_reference
        b = self.out['cut'].levels.ustar_reference
        self.assertEqual(a.scenario, b.scenario)
        self.assertEqual(a.scenarios, b.scenarios)
        pd.testing.assert_frame_equal(a.result.scores, b.result.scores)
        pd.testing.assert_series_equal(a.result.masked, b.result.masked)
        # The REF output loses the long gaps of its own scenario (NEE_CUT_REF_QC)
        mask = self._mask(b.scenario)
        self.assertGreater(int(mask.sum()), 0)
        self._check_column(b.gapfilled_col, mask)
        self._check_column(b.flag_col, mask)

    def test_partitioning_on_the_full_fill(self):
        full, cut = self.out['full'], self.out['cut']
        self.assertEqual(list(full.levels.level42_nt_of), list(cut.levels.level42_nt_of))
        for scen, inst in cut.levels.level42_nt_of.items():
            with self.subTest(scen=scen):
                # The instances partitioned the full fill and keep their results
                pd.testing.assert_frame_equal(inst.results,
                                              full.levels.level42_nt_of[scen].results)
                mask = self._mask(scen)
                for col in inst.results.columns:
                    name = f'{col}_{scen}'
                    if col.startswith(('GPP_', 'RECO_')):
                        self._check_column(name, mask)
                    else:
                        pd.testing.assert_series_equal(full.fpc_df[name], cut.fpc_df[name])

    def test_gpp_reco_ref(self):
        from diive.flux.lowres.ustar_reference import URE_PERCENTILES, oneflux_period_days
        a = self.out['full'].levels.partitioning_reference['nt_of']
        b = self.out['cut'].levels.partitioning_reference['nt_of']
        self.assertEqual(a.long_gaps, {})
        self.assertEqual(set(b.long_gaps), {'GPP', 'RECO'})
        index = self.out['cut'].fpc_df.index
        ref_mask = self._mask(self.out['cut'].levels.ustar_reference.scenario)
        pct = [f'{p:02d}' for p in URE_PERCENTILES]
        for prod in ('GPP', 'RECO'):
            ra, rb = a.results[prod], b.results[prod]
            # Same selection: ure works on the full partitioning
            self.assertEqual(a.scenarios[prod], b.scenarios[prod])
            pd.testing.assert_frame_equal(ra.scores, rb.scores)
            gaps = b.long_gaps[prod]
            for agg in ('HH', 'DD', 'WW', 'MM', 'YY'):
                ta, tb, g = ra.tables[agg], rb.tables[agg], gaps[agg]
                with self.subTest(prod=prod, agg=agg):
                    pd.testing.assert_frame_equal(ta.mask(g), tb)
                    self.assertFalse(g['SE'].any())
                    pd.testing.assert_series_equal(g['MEAN'], g[pct].any(axis=1),
                                                   check_names=False)
            # Half-hourly: REF and USTAR50 masks of their NEE scenarios
            hh = gaps['HH']
            np.testing.assert_array_equal(hh['REF'].reindex(index).to_numpy(), ref_mask)
            np.testing.assert_array_equal(hh['USTAR50'].reindex(index).to_numpy(),
                                          self._mask('CUT_50'))
            self.assertTrue(hh[pct].any().any())
            self._check_column(b.columns[prod], ref_mask)
            # Daily: the days that hold a removed record (TIMESTAMP_START)
            removed_days = set((index[ref_mask] - pd.Timedelta('15min')).normalize())
            days = oneflux_period_days(rb.years)['DD'][0]
            self.assertEqual(set(days[gaps['DD']['REF'].to_numpy()]), removed_days)
            months = oneflux_period_days(rb.years)['MM'][0]
            self.assertEqual(list(months[gaps['MM']['REF'].to_numpy()]),
                             [pd.Timestamp('2021-07-01')])
            self.assertTrue(gaps['YY']['REF'].all())


if __name__ == '__main__':
    unittest.main()
