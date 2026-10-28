"""Tests for the ONEFlux REF u* selection (diive.flux.lowres.ustar_reference) and
the chain steps around it: given thresholds, VUT and REF in run_level33_ustar_detection,
run_level41_ustar_reference and the FluxConfig options.

Bitwise parity with the compiled ONEFlux nee_proc is checked outside the test
suite, in diive-data/references/ustar_parity/ref/.
"""
import unittest

import numpy as np
import pandas as pd

from diive.flux.lowres.ustar_reference import (
    CO2TOC, ONEFLUX_REF_PERCENTILES, _daily_kernel, _group_kernel, _month_lengths,
    _week_lengths, oneflux_model_efficiency, oneflux_ref_mask, select_ustar_reference,
)


class TestModelEfficiency(unittest.TestCase):

    def test_scores_by_hand(self):
        # Version 0 = [1, 2, 3] (var 2/3), version 1 = [2, 4, 6] (var 8/3).
        # MSE(0, 1) = (1 + 4 + 9) / 3 = 14/3.
        # Score 0 = 1 + (1 - (14/3) / (2/3)) = 1 - 6 = -5
        # Score 1 = (1 - (14/3) / (8/3)) + 1 = -0.75 + 1 = 0.25
        idx, scores, mef = oneflux_model_efficiency(np.array([[1., 2.], [2., 4.], [3., 6.]]))
        self.assertEqual(idx, 1)
        np.testing.assert_allclose(scores, [-5.0, 0.25], rtol=1e-12)
        np.testing.assert_allclose(mef, [[1.0, -0.75], [-6.0, 1.0]], rtol=1e-12)

    def test_only_common_records_enter_the_mse(self):
        # Version 1 is missing at record 2: MSE(0, 1) uses records 0 and 1 only,
        # but each variance uses all values of its own version.
        v = np.array([[1., 2.], [2., 4.], [3., np.nan]])
        idx, scores, mef = oneflux_model_efficiency(v)
        var0, var1 = 2 / 3, 1.0          # [1, 2, 3] and [2, 4]
        mse = (1 + 4) / 2
        self.assertAlmostEqual(mef[1, 0], 1 - mse / var0)
        self.assertAlmostEqual(mef[0, 1], 1 - mse / var1)
        self.assertEqual(idx, int(np.argmax(scores)))

    def test_ties_go_to_the_lowest_percentile(self):
        v = np.tile(np.arange(10.)[:, None], (1, 4))
        idx, scores, _ = oneflux_model_efficiency(v)
        self.assertEqual(idx, 0)
        np.testing.assert_array_equal(scores, [4.0] * 4)

    def test_minus_9999_is_missing_and_truncated(self):
        # (int)value == -9999 in the C: -9999.5 counts as missing as well.
        a = np.array([[1., 1.], [2., -9999.5], [3., 3.]])
        b = np.array([[1., 1.], [2., np.nan], [3., 3.]])
        self.assertEqual(oneflux_model_efficiency(a)[0], oneflux_model_efficiency(b)[0])
        np.testing.assert_array_equal(oneflux_model_efficiency(a)[1],
                                      oneflux_model_efficiency(b)[1])

    def test_version_without_values_fails(self):
        idx, scores, _ = oneflux_model_efficiency(np.array([[1., np.nan], [2., np.nan]]))
        self.assertEqual(idx, -1)
        self.assertTrue(np.isnan(scores).all())


class TestMask(unittest.TestCase):

    def test_interior_of_a_long_run(self):
        qc = [0] * 5 + [3] * 10 + [0] * 5
        mask, blocks = oneflux_ref_mask(qc, mef_max_gap=2, mef_min_gap=3)
        # bad records 5-14: the first and last 2 stay, 7-12 are left out
        self.assertEqual(np.flatnonzero(mask).tolist(), list(range(7, 13)))
        self.assertEqual(blocks, [(7, 13, True)])

    def test_block_shorter_than_min_gap_is_kept(self):
        qc = [0] * 5 + [3] * 10 + [0] * 5
        mask, blocks = oneflux_ref_mask(qc, mef_max_gap=2, mef_min_gap=7)
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [(7, 13, False)])

    def test_run_from_the_second_record_touches_the_start(self):
        # The forward counter starts at max_gap + 1 at record 0, whatever its
        # quality (dataset.c:779), so a good first record does not reset it.
        qc = [0] + [3] * 6 + [0] * 5
        mask, _ = oneflux_ref_mask(qc, mef_max_gap=2, mef_min_gap=1)
        self.assertEqual(np.flatnonzero(mask).tolist(), [1, 2, 3, 4])

    def test_last_record_is_never_left_out(self):
        qc = [0] * 5 + [3] * 5
        mask, blocks = oneflux_ref_mask(qc, mef_max_gap=2, mef_min_gap=1)
        self.assertEqual(np.flatnonzero(mask).tolist(), [7, 8])
        self.assertEqual(blocks, [(7, 9, True)])

    def test_quality_up_to_mef_qc_and_unfilled_are_good(self):
        for qc in ([2] * 20, [np.nan] * 20):
            mask, blocks = oneflux_ref_mask(qc, mef_max_gap=2, mef_min_gap=1)
            self.assertFalse(mask.any())
        # mef_qc=3: no filter at all
        mask, blocks = oneflux_ref_mask([3] * 20, mef_qc=3, mef_max_gap=2, mef_min_gap=1)
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [])


class TestAggregation(unittest.TestCase):

    def test_daily_mean_times_co2toc_and_missing(self):
        hh = np.array([[1., 2., 3., 4.], [1., -9999., 3., 4.]])
        dd = _daily_kernel(hh, 2, CO2TOC)
        np.testing.assert_array_equal(dd[0], [1.5 * CO2TOC, 3.5 * CO2TOC])
        np.testing.assert_array_equal(dd[1], [-9999., 3.5 * CO2TOC])

    def test_weeks_months_years(self):
        self.assertEqual(_week_lengths([2020]).tolist(), [7] * 51 + [9])
        self.assertEqual(_week_lengths([2021]).tolist(), [7] * 51 + [8])
        self.assertEqual(_month_lengths([2020])[1], 29)
        self.assertEqual(_month_lengths([2021, 2020]).sum(), 731)
        dd = np.array([[1., 2., 3., 4., 5.]])
        np.testing.assert_array_equal(_group_kernel(dd, np.array([2, 3]), True), [[1.5, 4.]])
        np.testing.assert_array_equal(_group_kernel(dd, np.array([2, 3]), False), [[3., 12.]])
        dd[0, 3] = -9999.
        np.testing.assert_array_equal(_group_kernel(dd, np.array([2, 3]), True), [[1.5, -9999.]])


def _versions(years, freq='30min', seed=0):
    """40 noisy versions of one signal; noise grows with the percentile."""
    idx = pd.date_range(f'{years[0]}-01-01', f'{years[-1] + 1}-01-01', freq=freq,
                        inclusive='left') + pd.Timedelta(freq) / 2
    rng = np.random.default_rng(seed)
    base = 5 * np.sin(np.arange(len(idx)) / (len(idx) / len(years) / 365) * 2 * np.pi)
    nee = pd.DataFrame({p: base + rng.normal(0, 1 + 0.01 * i, len(idx))
                        for i, p in enumerate(ONEFLUX_REF_PERCENTILES)}, index=idx)
    qc = pd.DataFrame(0.0, index=idx, columns=nee.columns)
    return nee, qc


class TestSelectUstarReference(unittest.TestCase):

    def test_single_year_yearly_ref_is_the_monthly_ref(self):
        nee, qc = _versions([2021])
        r = select_ustar_reference(nee, qc)
        self.assertEqual(r.index['YY'], r.index['MM'])
        self.assertTrue(r.scores.loc['YY'].isna().all())
        self.assertNotIn('YY', r.model_efficiency)
        for agg in ('HH', 'DD', 'WW', 'MM'):
            self.assertEqual(r.percentile[agg], ONEFLUX_REF_PERCENTILES[r.index[agg]])
            self.assertEqual(r.index[agg], int(np.argmax(r.scores.loc[agg].to_numpy())))

    def test_least_noisy_version_wins_on_records(self):
        nee, qc = _versions([2020, 2021])
        r = select_ustar_reference(nee, qc)
        self.assertEqual(r.percentile['HH'], 1.25)
        self.assertIn('YY', r.model_efficiency)

    def test_long_low_quality_gap_is_left_out_in_all_versions(self):
        nee, qc = _versions([2021])
        qc.iloc[1000:4000, -1] = 3      # only the highest percentile's QC counts
        qc.iloc[6000:9000, 0] = 3
        r = select_ustar_reference(nee, qc)
        self.assertEqual(int(r.masked.sum()), 3000 - 2 * 720)
        self.assertTrue(r.masked.iloc[1720:3280].all())
        self.assertEqual(len(r.masked_blocks), 1)

    def test_partial_year_is_padded(self):
        nee, qc = _versions([2021])
        r_full = select_ustar_reference(nee, qc)
        r_part = select_ustar_reference(nee.iloc[100:], qc.iloc[100:])
        self.assertEqual(len(r_part.masked), len(r_full.masked))
        # Day 1 is incomplete: missing in every version, so DD scores differ
        self.assertFalse(np.array_equal(r_part.scores.loc['DD'], r_full.scores.loc['DD']))

    def test_hourly(self):
        nee, qc = _versions([2021], freq='60min')
        r = select_ustar_reference(nee, qc)
        self.assertEqual(len(r.masked), 8760)

    def test_input_checks(self):
        nee, qc = _versions([2021])
        with self.assertRaises(ValueError):
            select_ustar_reference(nee.iloc[:, :3], qc)
        with self.assertRaises(ValueError):
            select_ustar_reference(nee, qc, percentiles=[1, 2])
        n3 = nee.resample('3h').mean()
        with self.assertRaises(ValueError):
            select_ustar_reference(n3, n3 * 0)


def _chain_to_l32():
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    from diive.flux.postprocessingchain import (
        init_flux_data, make_level32_detector, run_level2, run_level31, run_level32,
    )
    df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
    for c in ('TA_1_1_1', 'SW_IN_1_1_1'):
        df[c] = df[c].bfill()
    df['VPD_kPa'] = df['VPD_EP'].bfill() * 0.1
    data = init_flux_data(df=df, fluxcol='FC', site_lat=46.58, site_lon=9.79, utc_offset=1)
    data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
    data = run_level31(data)
    data, sod = make_level32_detector(data)
    sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
    sod.addflag()
    return run_level32(data, outlier_detector=sod)


def _ref_thresholds():
    thr = {16: 0.10, 50: 0.15, 84: 0.20}
    thr.update({p: 0.05 + 0.002 * p for p in ONEFLUX_REF_PERCENTILES})
    return thr


class TestLevel33GivenThresholds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _chain_to_l32()

    def test_given_dict_equals_constant(self):
        from diive.flux.postprocessingchain import (
            run_level33_constant_ustar, run_level33_ustar_detection,
        )
        a = run_level33_ustar_detection(self.data, thresholds={16: 0.1, 50: 0.15, 84: 0.2},
                                        showplot=False, verbose=False)
        b = run_level33_constant_ustar(self.data, thresholds=[0.1, 0.15, 0.2],
                                       threshold_labels=['CUT_16', 'CUT_50', 'CUT_84'],
                                       showplot=False, verbose=False)
        pd.testing.assert_frame_equal(a.fpc_df, b.fpc_df)
        self.assertEqual(list(a.levels.level33_qcf), ['CUT_16', 'CUT_50', 'CUT_84'])
        self.assertIsNone(a.levels.ustar_detection)
        rec = a.levels.ustar_scenarios
        self.assertEqual((rec.source, rec.threshold_type), ('given', 'cut'))
        self.assertEqual(rec.thresholds, {'CUT_16': 0.1, 'CUT_50': 0.15, 'CUT_84': 0.2})
        self.assertEqual(rec.reference_labels, ())
        # A list aligned with percentiles gives the same
        c = run_level33_ustar_detection(self.data, thresholds=[0.1, 0.15, 0.2],
                                        showplot=False, verbose=False)
        pd.testing.assert_frame_equal(a.fpc_df, c.fpc_df)
        # The lower-level call does not leave a record
        self.assertIsNone(b.levels.ustar_scenarios)

    def test_given_vut_per_year_equals_variable(self):
        from diive.flux.postprocessingchain import (
            run_level33_ustar_detection, run_level33_variable_ustar,
        )
        a = run_level33_ustar_detection(self.data, mode='vut',
                                        thresholds={2021: {16: 0.1, 50: 0.15}},
                                        showplot=False, verbose=False)
        idx = self.data.fpc_df.index
        b = run_level33_variable_ustar(
            self.data, threshold_series={'VUT_16': pd.Series(0.1, index=idx),
                                         'VUT_50': pd.Series(0.15, index=idx)},
            showplot=False, verbose=False)
        pd.testing.assert_frame_equal(a.fpc_df, b.fpc_df)
        self.assertEqual(a.levels.ustar_scenarios.thresholds['VUT_50'], {2021: 0.15})

    def test_forms_that_do_not_fit_mode_raise(self):
        from diive.flux.postprocessingchain import run_level33_ustar_detection
        kw = dict(showplot=False, verbose=False)
        with self.assertRaisesRegex(ValueError, "mode='vut' needs"):
            run_level33_ustar_detection(self.data, mode='vut', thresholds={50: 0.1}, **kw)
        with self.assertRaisesRegex(ValueError, "pass mode='vut'"):
            run_level33_ustar_detection(self.data, thresholds={2021: {50: 0.1}}, **kw)
        with self.assertRaisesRegex(ValueError, 'no values for year'):
            run_level33_ustar_detection(self.data, mode='vut', thresholds={2020: {50: 0.1}}, **kw)
        with self.assertRaisesRegex(ValueError, 'aligned with'):
            run_level33_ustar_detection(self.data, thresholds=[0.1, 0.2], **kw)
        with self.assertRaisesRegex(ValueError, '40 ONEFlux percentiles'):
            run_level33_ustar_detection(self.data, thresholds={50: 0.1}, reference=True, **kw)
        with self.assertRaisesRegex(ValueError, 'ta_col and swin_col'):
            run_level33_ustar_detection(self.data, **kw)

    def test_reference_step(self):
        from diive.flux.postprocessingchain import (
            run_level33_ustar_detection, run_level41_mds, run_level41_ustar_reference,
        )
        d = run_level33_ustar_detection(self.data, thresholds=_ref_thresholds(),
                                        reference=True, showplot=False, verbose=False)
        labels = list(d.levels.level33_qcf)
        self.assertEqual(labels[:3], ['CUT_16', 'CUT_50', 'CUT_84'])
        self.assertEqual(len(labels), 43)
        rec = d.levels.ustar_scenarios
        self.assertEqual(len(rec.reference_labels), 40)
        self.assertEqual(rec.percentiles['CUT_0125'], 1.25)
        with self.assertRaisesRegex(RuntimeError, 'run_level41_mds'):
            run_level41_ustar_reference(d, verbose=False)

        d = run_level41_mds(d, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa', mode='oneflux')
        d = run_level41_ustar_reference(d, verbose=False)
        ref = d.levels.ustar_reference
        self.assertEqual(ref.label, 'CUT_REF')
        self.assertIn(ref.scenario, rec.reference_labels)
        self.assertEqual(ref.percentile, rec.percentiles[ref.scenario])
        self.assertEqual(ref.threshold, rec.thresholds[ref.scenario])
        self.assertEqual(ref.gapfilled_col, 'NEE_L3.1_L3.3_CUT_REF_QCF_gfMDS')
        src = d.gapfilled_cols()['mds'][ref.scenario]
        pd.testing.assert_series_equal(d.fpc_df[ref.gapfilled_col], d.fpc_df[src],
                                       check_names=False)
        self.assertEqual(set(ref.scenarios), {'HH', 'DD', 'WW', 'MM', 'YY'})
        self.assertEqual(ref.scenarios['HH'], ref.scenario)
        # The other scenarios are unchanged and REF is not an L4.1 scenario
        self.assertNotIn('CUT_REF', d.gapfilled_cols()['mds'])

        # Slow L4.1/L4.2 steps run on the requested scenarios plus the REF one,
        # not on the 40 REF versions; an explicit list overrides that.
        from diive.flux.postprocessingchain.levels._shared import select_scenarios
        self.assertEqual(rec.main_labels, ('CUT_16', 'CUT_50', 'CUT_84'))
        before_ref = run_level41_mds(
            run_level33_ustar_detection(self.data, thresholds=_ref_thresholds(),
                                        reference=True, showplot=False, verbose=False),
            swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa', mode='oneflux')
        self.assertEqual(list(select_scenarios(before_ref)), ['CUT_16', 'CUT_50', 'CUT_84'])
        self.assertEqual(list(select_scenarios(d)),
                         [lab for lab in labels
                          if lab in ('CUT_16', 'CUT_50', 'CUT_84', ref.scenario)])
        self.assertEqual(list(select_scenarios(d, ['CUT_0125', 'CUT_50'])),
                         ['CUT_0125', 'CUT_50'])
        with self.assertRaisesRegex(KeyError, 'Unknown USTAR scenario'):
            select_scenarios(d, ['CUT_99'])

        # Re-running MDS drops the stale REF
        d2 = run_level41_mds(d, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa', mode='oneflux')
        self.assertIsNone(d2.levels.ustar_reference)
        self.assertNotIn(ref.gapfilled_col, d2.fpc_df.columns)
        # Re-running L3.3 drops it too
        d3 = run_level33_ustar_detection(d, thresholds={50: 0.15}, showplot=False, verbose=False)
        self.assertIsNone(d3.levels.ustar_reference)
        self.assertNotIn(ref.gapfilled_col, d3.fpc_df.columns)
        self.assertEqual(d3.levels.ustar_scenarios.reference_labels, ())


class TestRunChainUstarOptions(unittest.TestCase):

    @staticmethod
    def _data():
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import init_flux_data
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        for c in ('TA_1_1_1', 'SW_IN_1_1_1'):
            df[c] = df[c].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill() * 0.1
        return init_flux_data(df=df, fluxcol='FC', site_lat=46.58, site_lon=9.79, utc_offset=1)

    @staticmethod
    def _cfg(**kw):
        from diive.flux.postprocessingchain import FluxConfig
        base = dict(fluxcol='FC',
                    level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
                    gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,
                    mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa')
        base.update(kw)
        return FluxConfig(**base)

    def test_percentile_thresholds_vut_and_reference(self):
        from diive.flux.postprocessingchain import run_chain
        data = self._data()
        # Percentile-keyed CUT: labels as detected ones
        out = run_chain(data, self._cfg(ustar_thresholds={16: 0.1, 50: 0.15}))
        self.assertEqual(list(out.levels.level33_qcf), ['CUT_16', 'CUT_50'])
        self.assertEqual(out.levels.ustar_scenarios.source, 'given')
        # Given VUT per year
        out = run_chain(data, self._cfg(ustar_thresholds={2021: {50: 0.15}},
                                        ustar_threshold_type='vut'))
        self.assertEqual(list(out.levels.level33_qcf), ['VUT_50'])
        # REF
        out = run_chain(data, self._cfg(ustar_thresholds=_ref_thresholds(),
                                        ustar_reference=True, mds_mode='oneflux'))
        ref = out.levels.ustar_reference
        self.assertEqual(ref.label, 'CUT_REF')
        self.assertIn(ref.gapfilled_col, out.fpc_df.columns)

    def test_option_checks(self):
        from diive.flux.postprocessingchain import run_chain
        data = self._data()
        with self.assertRaisesRegex(ValueError, 'ustar_threshold_type'):
            run_chain(data, self._cfg(ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
                                      ustar_threshold_type='x'))
        with self.assertRaisesRegex(ValueError, 'per year'):
            run_chain(data, self._cfg(ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
                                      ustar_threshold_type='vut'))
        with self.assertRaisesRegex(ValueError, 'keyed by percentile'):
            run_chain(data, self._cfg(ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
                                      ustar_reference=True))
        with self.assertRaisesRegex(ValueError, 'gapfill_mds=True'):
            run_chain(data, self._cfg(ustar_thresholds=_ref_thresholds(), ustar_reference=True,
                                      gapfill_mds=False))
        # RF/XGB/L4.2 run only on the main scenarios plus REF now, so REF with
        # them on no longer warns (it still checks the feature columns).
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*40 u\\* scenarios.*')
            with self.assertRaises(KeyError):  # unknown RF feature
                run_chain(data, self._cfg(ustar_thresholds=_ref_thresholds(),
                                          ustar_reference=True, gapfill_rf=True,
                                          gapfilling_features=['NOT_THERE']))

    def test_bootstrap_vut_with_reference(self):
        """Detection in run_chain: VUT per year at the 40 REF percentiles plus 50."""
        from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
        from diive.flux.postprocessingchain import FluxConfig, init_flux_data, run_chain
        df = load_exampledata_parquet_lae_level1_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        data = init_flux_data(df=df, fluxcol='FC', site_lat=47.48, site_lon=8.36, utc_offset=1)
        cfg = FluxConfig(
            fluxcol='FC', ustar_detection_mode='bootstrap',
            ustar_bootstrap_detection_mode='oneflux_mp', ustar_bootstrap_n_iter=5,
            ustar_bootstrap_ta_col='TA_T1_47_1_gfXG', ustar_bootstrap_swin_col='SW_IN_T1_47_1_gfXG',
            ustar_bootstrap_percentiles=(50,), ustar_threshold_type='vut', ustar_reference=True,
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True, mds_mode='oneflux',
            mds_swin='SW_IN_T1_47_1_gfXG', mds_ta='TA_T1_47_1_gfXG', mds_vpd='VPD_T1_47_1_gfXG')
        out = run_chain(data, cfg)
        rec = out.levels.ustar_scenarios
        self.assertEqual((rec.source, rec.threshold_type), ('detected', 'vut'))
        self.assertIsNotNone(out.levels.ustar_detection)
        self.assertEqual(list(out.levels.level33_qcf)[0], 'VUT_50')
        ref = out.levels.ustar_reference
        self.assertEqual(ref.label, 'VUT_REF')
        self.assertEqual(set(ref.threshold), {2024})
        self.assertEqual(ref.result.index['YY'], ref.result.index['MM'])  # single year


# ---------------------------------------------------------------- GPP / RECO (ure)

def _ure_versions(years, freq='30min'):
    """40 versions: version i = base + i, base alternating 0 and 2.

    All sums are exact: every version has mean base + i with base mean 1 and
    variance 1, MSE(j, i) = (i - j)^2, so score j = 40 - sum_i (i - j)^2. The
    best are 19 and 20 (48.75, 51.25), the tie goes to 19.
    """
    idx = pd.date_range(f'{years[0]}-01-01', f'{years[-1] + 1}-01-01', freq=freq,
                        inclusive='left') + pd.Timedelta(freq) / 2
    idx.name = 'TIMESTAMP_MIDDLE'
    base = np.tile([0.0, 2.0], len(idx) // 2)
    v = pd.DataFrame({p: base + i for i, p in enumerate(ONEFLUX_REF_PERCENTILES)}, index=idx)
    return v, pd.Series(base + 19.5, index=idx)


class TestUreRefMatrix(unittest.TestCase):

    def test_valid_values_threshold_in_float(self):
        from diive.flux.lowres.ustar_reference import _ure_ref_matrix
        # (int)(0.7f * 90) is 63 in float, 62 in double ("0.7 * 90 = 62 instead
        # of 63", ure dataset.c:411): a version with 62 valid values drops out.
        m = np.zeros((90, 12))
        m[62:, 0] = -9999.
        red, versions = _ure_ref_matrix(m, 70, 10)
        self.assertEqual(versions.tolist(), list(range(1, 12)))
        self.assertEqual(red.shape, (90, 11))
        m[62, 0] = 0.  # 63 valid: it takes part, the rows it misses go
        red, versions = _ure_ref_matrix(m, 70, 10)
        self.assertEqual(versions.tolist(), list(range(12)))
        self.assertEqual(red.shape, (63, 12))

    def test_rows_missing_everywhere_do_not_count(self):
        from diive.flux.lowres.ustar_reference import _ure_ref_matrix
        m = np.ones((10, 10))
        m[5:] = -9999.
        m[4, 0] = -9999.
        # rows 5-9 go first: n_max = 5, need (int)(0.9f * 5) = 4, version 0 has 4
        red, versions = _ure_ref_matrix(m, 90, 10)
        self.assertEqual(len(versions), 10)
        self.assertEqual(red.shape, (4, 10))

    def test_too_few_versions_or_rows(self):
        from diive.flux.lowres.ustar_reference import _ure_ref_matrix
        self.assertIsNone(_ure_ref_matrix(np.ones((10, 9)), 90, 10))
        self.assertIsNotNone(_ure_ref_matrix(np.ones((10, 10)), 90, 10))
        self.assertIsNone(_ure_ref_matrix(np.full((10, 10), -9999.), 90, 10))

    def test_percentile_positions(self):
        from diive.flux.lowres.ustar_reference import _ure_percentile_index
        # ROUND(p / 100 * n) - 1: 0.05 * 40 = 2 -> 1, 0.16 * 40 = 6.4 -> 5,
        # 0.84 * 40 = 33.6 -> 33, 0.5 * 15 = 7.5 -> 7, 0.25 * 15 = 3.75 -> 3
        self.assertEqual([_ure_percentile_index(p, 40) for p in (5, 16, 25, 50, 75, 84, 95)],
                         [1, 5, 9, 19, 29, 33, 37])
        self.assertEqual([_ure_percentile_index(p, 15) for p in (25, 50, 75)], [3, 7, 10])
        self.assertEqual(_ure_percentile_index(5, 1), 0)


class TestSelectPartitioningReference(unittest.TestCase):

    def test_ref_and_statistics_by_hand(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        r = select_partitioning_reference(v, ust50=u50)
        self.assertEqual(r.percentile['HH'], 48.75)
        self.assertEqual(r.scores.loc['HH', 48.75], r.scores.loc['HH', 51.25])
        self.assertEqual(r.scores.loc['HH', 48.75], 40 - sum((i - 19) ** 2 for i in range(40)))
        hh = r.tables['HH']
        base = v[1.25].to_numpy()
        np.testing.assert_array_equal(hh['REF'], base + 19)
        np.testing.assert_array_equal(hh['USTAR50'], base + 19.5)
        np.testing.assert_array_equal(hh['MEAN'], base + 19.5)
        for col, i in zip(('05', '16', '25', '50', '75', '84', '95'), (1, 5, 9, 19, 29, 33, 37),
                          strict=True):
            np.testing.assert_array_equal(hh[col], base + i)
        np.testing.assert_allclose(hh['SE'], np.std(np.arange(40.), ddof=1) / np.sqrt(40),
                                   rtol=1e-14)
        # Single year, VUT: yearly REF = monthly REF
        self.assertEqual(r.index['YY'], r.index['MM'])
        self.assertTrue(r.scores.loc['YY'].isna().all())

    def test_few_valid_versions_give_fewer_statistics(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        day1 = v.index.dayofyear == 10
        day2 = v.index.dayofyear == 11
        v.loc[day1, v.columns[:25]] = np.nan  # 15 valid
        v.loc[day2, v.columns[:32]] = np.nan  # 8 valid
        r = select_partitioning_reference(v, ust50=u50)
        hh = r.tables['HH']
        base = v[98.75].to_numpy() - 39
        sel = hh.loc[day1]
        for col in ('05', '16', '84', '95'):
            self.assertTrue(sel[col].isna().all())
        # sorted versions 25-39: positions 3, 7, 10
        np.testing.assert_array_equal(sel['25'], base[day1] + 28)
        np.testing.assert_array_equal(sel['50'], base[day1] + 32)
        np.testing.assert_array_equal(sel['75'], base[day1] + 35)
        np.testing.assert_array_equal(sel['MEAN'], base[day1] + 32)
        self.assertTrue(hh.loc[day2].drop(columns=['REF', 'USTAR50']).isna().all().all())
        # Rows with a missing version leave the selection
        self.assertEqual(r.records_used['HH'], 17520 - 96)
        self.assertEqual(len(r.versions_used['HH']), 40)

    def test_version_with_fewer_values_is_left_out(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        v.loc[v.index.month <= 2, 3.75] = np.nan  # 59 of 365 days missing: 84 %
        r = select_partitioning_reference(v, ust50=u50)
        self.assertNotIn(3.75, r.versions_used['HH'])
        self.assertEqual(len(r.versions_used['HH']), 39)
        self.assertTrue(np.isnan(r.scores.loc['HH', 3.75]))
        self.assertEqual(r.records_used['HH'], 17520)

    def test_week_52_starts_after_the_valid_days_of_ustar50(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        r = select_partitioning_reference(v, ust50=u50)
        self.assertEqual(r.tables['WW'].index[-1], pd.Timestamp('2021-12-24'))
        # USTAR50 misses 2021-12-18 (day 352): one valid day in week 51
        gap = u50.copy()
        gap[gap.index.normalize() == pd.Timestamp('2021-12-18')] = np.nan
        r = select_partitioning_reference(v, ust50=gap)
        ww = r.tables['WW']
        self.assertEqual(ww.index[-1], pd.Timestamp('2021-12-18'))
        self.assertAlmostEqual(ww['MEAN'].iloc[-1], r.tables['DD']['MEAN'].iloc[351:].mean(),
                               places=12)
        # ust50_week decides, not ust50
        r = select_partitioning_reference(v, ust50=gap, ust50_week=u50)
        self.assertEqual(r.tables['WW'].index[-1], pd.Timestamp('2021-12-24'))
        # Without USTAR50, ure sees it missing: week 52 starts at day 351
        r = select_partitioning_reference(v)
        self.assertEqual(r.tables['WW'].index[-1], pd.Timestamp('2021-12-17'))
        self.assertTrue(r.tables['HH']['USTAR50'].isna().all())

    def test_yearly_rule_vut_and_cut(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        rng = np.random.default_rng(1)
        idx = pd.date_range('2019-01-01', '2022-01-01', freq='30min',
                            inclusive='left') + pd.Timedelta('15min')
        v = pd.DataFrame({p: 5 + rng.normal(0, 1 + 0.01 * i, len(idx))
                          for i, p in enumerate(ONEFLUX_REF_PERCENTILES)}, index=idx)
        # 2019 and 2021 incomplete in every version: one complete year
        v.iloc[100] = np.nan
        v.iloc[-100] = np.nan
        vut = select_partitioning_reference(v, threshold_type='vut')
        cut = select_partitioning_reference(v, threshold_type='cut')
        self.assertEqual(vut.records_used['YY'], 1)
        self.assertEqual(vut.index['YY'], vut.index['MM'])
        # ure has no such rule for CUT: NaN scores, the first version
        self.assertEqual(cut.index['YY'], 0)
        self.assertTrue(cut.scores.loc['YY'].isna().all())
        self.assertIn('YY', cut.model_efficiency)
        # CUT with fewer than 3 years (no ONEFlux CUT): diive applies the rule
        one = select_partitioning_reference(v.loc['2020'], threshold_type='cut')
        self.assertEqual(one.index['YY'], one.index['MM'])

    def test_mask_leaves_out_the_versions_but_not_ustar50(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        mask = pd.Series(False, index=v.index)
        mask.iloc[1000:1500] = True
        r = select_partitioning_reference(v, ust50=u50, mask=mask)
        hh = r.tables['HH']
        self.assertTrue(hh['REF'].iloc[1000:1500].isna().all())
        self.assertTrue(hh['MEAN'].iloc[1000:1500].isna().all())
        self.assertFalse(hh['USTAR50'].iloc[1000:1500].isna().any())
        self.assertEqual(int(r.masked.sum()), 500)
        self.assertEqual(r.records_used['HH'], 17520 - 500)
        # Days touched by the mask are missing in every version
        self.assertTrue(r.tables['DD']['REF'].iloc[20:32].isna().all())
        self.assertFalse(r.tables['DD']['USTAR50'].iloc[20:32].isna().any())

    def test_timestamp_end_index_and_checks(self):
        from diive.flux.lowres.ustar_reference import select_partitioning_reference
        v, u50 = _ure_versions([2021])
        r_mid = select_partitioning_reference(v, ust50=u50)
        v_end = v.copy()
        v_end.index = v.index + pd.Timedelta('15min')
        v_end.index.name = 'TIMESTAMP_END'
        r_end = select_partitioning_reference(v_end, ust50=u50.set_axis(v_end.index))
        pd.testing.assert_frame_equal(r_mid.tables['HH'], r_end.tables['HH'])
        with self.assertRaises(ValueError):
            select_partitioning_reference(v.iloc[:, :3])
        with self.assertRaises(ValueError):
            select_partitioning_reference(v, ust50=u50.iloc[1:])
        with self.assertRaises(ValueError):
            select_partitioning_reference(v, threshold_type='x')


class TestUreCleanup(unittest.TestCase):

    def test_nighttime(self):
        from diive.flux.lowres.ustar_reference import oneflux_ure_cleanup
        reco = np.array([1., np.nan, 150., -150., 2., 3., -9999.5])
        gpp = np.array([1., 1., 1., 1., np.inf, -7000., 1.])
        r, g = oneflux_ure_cleanup(reco, gpp, 'nighttime')
        # missing or out-of-range RECO removes GPP as well, not the other way
        np.testing.assert_array_equal(r, [1., np.nan, np.nan, np.nan, 2., 3., np.nan])
        np.testing.assert_array_equal(g, [1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    def test_daytime(self):
        from diive.flux.lowres.ustar_reference import oneflux_ure_cleanup
        reco = pd.Series([1., 150., np.nan, 2., 3.])
        gpp = pd.Series([-0.1, 1., 1., 200., 0.])
        r, g = oneflux_ure_cleanup(reco, gpp, 'daytime')
        self.assertIsInstance(r, pd.Series)
        np.testing.assert_array_equal(r, [1., np.nan, np.nan, 2., 3.])
        np.testing.assert_array_equal(g, [np.nan, 1., 1., np.nan, 0.])
        with self.assertRaises(ValueError):
            oneflux_ure_cleanup(reco, gpp, 'x')


def _chain_to_ref():
    """Example data (one month) to the NEE REF, plus nighttime ONEFlux at L4.2."""
    from diive.flux.postprocessingchain import (
        run_level33_ustar_detection, run_level41_mds, run_level41_ustar_reference,
        run_level42_partitioning,
    )
    d = run_level33_ustar_detection(_chain_to_l32(), thresholds=_ref_thresholds(),
                                    reference=True, showplot=False, verbose=False)
    d = run_level41_mds(d, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa', mode='oneflux')
    d = run_level41_ustar_reference(d, verbose=False)
    return run_level42_partitioning(d, method='nighttime', ta='TA_1_1_1', sw_in='SW_IN_1_1_1',
                                    ta_f='TA_1_1_1', verbose=0)


_NT = dict(ta='TA_1_1_1', sw_in='SW_IN_1_1_1', ta_f='TA_1_1_1', verbose=0)


class TestLevel42UstarReference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from diive.flux.postprocessingchain import run_level42_ustar_reference
        cls.before = _chain_to_ref()
        cls.data = run_level42_ustar_reference(cls.before, method='nighttime', **_NT)

    def test_needs_mds_filled_versions(self):
        """L417: a gap-filling method that did not fill the 40 versions is refused up front."""
        from diive.flux.postprocessingchain import run_level42_ustar_reference
        for method in ('nighttime', 'daytime'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "gapfill_method='mds'"):
                    run_level42_ustar_reference(self.before, method=method, gapfill_method='rf',
                                                sw_in_f='SW_IN_1_1_1', vpd_f='VPD_kPa', **_NT)

    def test_needs_the_nee_ref(self):
        from diive.flux.postprocessingchain import (
            run_level33_ustar_detection, run_level42_ustar_reference,
        )
        d = run_level33_ustar_detection(_chain_to_l32(), thresholds={50: 0.15},
                                        showplot=False, verbose=False)
        with self.assertRaisesRegex(RuntimeError, 'run_level41_ustar_reference'):
            run_level42_ustar_reference(d, method='nighttime', **_NT)

    def test_ref_columns_and_record(self):
        from diive.flux.lowres.ustar_reference import oneflux_ure_cleanup
        d = self.data
        rec = d.levels.partitioning_reference['nt_of']
        self.assertEqual((rec.variant, rec.label), ('nt_of', 'CUT_REF'))
        self.assertEqual(rec.columns, {'GPP': 'GPP_NT_OF_CUT_REF', 'RECO': 'RECO_NT_OF_CUT_REF'})
        self.assertEqual(d.added_columns['L4.2_ref_nt_of'], list(rec.columns.values()))
        scen = d.levels.ustar_scenarios
        self.assertEqual(set(rec.instances), set(scen.reference_labels) | {'CUT_50'})
        for prod, pos in (('GPP', 1), ('RECO', 0)):
            source = rec.scenarios[prod]['HH']
            self.assertIn(source, scen.reference_labels)
            self.assertEqual(rec.results[prod].percentile['HH'], scen.percentiles[source])
            res = rec.instances[source].results
            expected = oneflux_ure_cleanup(res['RECO_NT_OF'], res['GPP_NT_OF'], 'nighttime')[pos]
            pd.testing.assert_series_equal(d.fpc_df[rec.columns[prod]], expected,
                                           check_names=False)
            self.assertEqual(set(rec.results[prod].tables), {'HH', 'DD', 'WW', 'MM', 'YY'})
        # The variant's own L4.2 results are not changed
        self.assertEqual(list(d.levels.level42_nt_of), list(self.before.levels.level42_nt_of))
        pd.testing.assert_frame_equal(d.fpc_df[self.before.fpc_df.columns], self.before.fpc_df)

    def test_rerun_replaces_and_variants_add_up(self):
        from diive.flux.postprocessingchain import run_level42_ustar_reference
        d = run_level42_ustar_reference(self.data, method='nighttime', **_NT)
        self.assertEqual(list(d.fpc_df.columns), list(self.data.fpc_df.columns))
        d = run_level42_ustar_reference(d, method='nighttime', mode='reddyproc', **_NT)
        self.assertEqual(set(d.levels.partitioning_reference), {'nt_of', 'nt_rp'})
        self.assertIn('GPP_NT_RP_CUT_REF', d.fpc_df.columns)
        self.assertIn('GPP_NT_OF_CUT_REF', d.fpc_df.columns)

    def test_upstream_reruns_clear_it(self):
        from diive.flux.postprocessingchain import run_level33_ustar_detection, run_level41_mds
        d = run_level41_mds(self.data, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa',
                            mode='oneflux')
        self.assertEqual(d.levels.partitioning_reference, {})
        self.assertNotIn('GPP_NT_OF_CUT_REF', d.fpc_df.columns)
        self.assertNotIn('L4.2_ref_nt_of', d.added_columns)
        d = run_level33_ustar_detection(self.data, thresholds={50: 0.15}, showplot=False,
                                        verbose=False)
        self.assertEqual(d.levels.partitioning_reference, {})
        self.assertNotIn('RECO_NT_OF_CUT_REF', d.fpc_df.columns)

    def test_run_chain_option(self):
        from diive.flux.postprocessingchain import run_chain
        cfg = TestRunChainUstarOptions._cfg
        data = TestRunChainUstarOptions._data()
        kw = dict(ustar_thresholds=_ref_thresholds(), mds_mode='oneflux',
                  partition_method='nighttime', partition_ta='TA_1_1_1',
                  partition_sw_in='SW_IN_1_1_1', partition_ta_f='TA_1_1_1')
        with self.assertRaisesRegex(ValueError, 'ustar_reference=True'):
            run_chain(data, cfg(**kw, partition_reference=True))
        with self.assertRaisesRegex(ValueError, 'a partitioning variant'):
            run_chain(data, cfg(ustar_thresholds=_ref_thresholds(), ustar_reference=True,
                                partition_reference=True))
        out = run_chain(data, cfg(**kw, ustar_reference=True, partition_reference=True))
        rec = out.levels.partitioning_reference['nt_of']
        self.assertEqual(rec.label, 'CUT_REF')
        for col in rec.columns.values():
            self.assertGreater(out.fpc_df[col].notna().sum(), 0)
        # L4.2 itself still runs on the main scenarios plus the NEE REF one
        self.assertNotIn('CUT_0125', out.levels.level42_nt_of)


if __name__ == '__main__':
    unittest.main()
