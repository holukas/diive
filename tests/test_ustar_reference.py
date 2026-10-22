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


if __name__ == '__main__':
    unittest.main()
