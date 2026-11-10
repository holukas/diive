"""Tests of the long-term XGBoost gap-filling with a Lloyd-Taylor fill of the wet night gaps."""
import unittest

import numpy as np
import pandas as pd

from diive.flux.partitioning.nighttime_oneflux import lloyd_taylor
from diive.gapfilling import LongTermGapFillingXGBoostTS, LongTermGapFillingXGBoostWetNightTS
from diive.gapfilling.wetnight import FLAG_WETNIGHT

XGB = dict(n_estimators=40, max_depth=3, learning_rate=0.3, random_state=42)
E0, RREF = 200.0, 4.0


def _synthetic(years=(2020, 2021, 2022), seed=1):
    """Night NEE from Lloyd-Taylor, day NEE from light; wet nights are gaps."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(f'{years[0]}-01-01 00:15', f'{years[-1]}-12-31 23:45', freq='30min',
                          name='TIMESTAMP_MIDDLE')
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    ta = 8 - 8 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    sw = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    night = pd.Series(sw == 0, index=index)
    nee = np.where(night, lloyd_taylor(ta, RREF, E0), lloyd_taylor(ta, RREF, E0) - 0.02 * sw)
    nee = nee + rng.normal(0, 0.3, len(index))
    wet = pd.Series(rng.random(len(index) // 48 + 1)[np.arange(len(index)) // 48] < 0.15, index=index)
    target = pd.Series(nee, index=index, name='NEE')
    target[(night & wet).to_numpy()] = np.nan               # the analyser misses the wet nights
    target[rng.random(len(index)) < 0.1] = np.nan           # and some other records
    df = pd.DataFrame({'NEE': target, 'TA': ta, 'SW_IN': sw}, index=index)
    return df, night, wet, pd.Series(ta, index=index)


class TestWetNightGapFilling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.night, cls.wet, cls.ta = _synthetic()
        cls.plain = LongTermGapFillingXGBoostTS(input_df=cls.df, target_col='NEE', **XGB).run()
        cls.wn = LongTermGapFillingXGBoostWetNightTS(input_df=cls.df, target_col='NEE', wet=cls.wet, ta=cls.ta,
                                                     night=cls.night, **XGB).run()

    def test_names(self):
        self.assertEqual(self.wn.gapfilled_.name, 'NEE_gfXGWN')
        self.assertEqual(self.wn.get_flag().name, 'FLAG_NEE_gfXGWN_ISFILLED')
        flags = [c for c in self.wn.gapfilling_df_.columns if str(c).startswith('FLAG_') and str(c).endswith('_ISFILLED')]
        self.assertEqual(flags, ['FLAG_NEE_gfXGWN_ISFILLED'])

    def test_flag_marks_the_wet_night_gaps(self):
        flag = self.wn.get_flag()
        gap = self.df['NEE'].isna()
        expected = gap & self.night & self.wet
        self.assertTrue(((flag == FLAG_WETNIGHT) == expected).all())
        self.assertEqual(self.wn.n_wetnight_filled_, int(expected.sum()))
        self.assertTrue((flag[~gap] == 0).all())

    def test_measured_and_other_gaps_unchanged(self):
        other = self.wn.get_flag() != FLAG_WETNIGHT
        np.testing.assert_allclose(self.wn.gapfilled_[other].to_numpy(), self.plain.gapfilled_[other].to_numpy())
        measured = self.df['NEE'].notna()
        np.testing.assert_allclose(self.wn.gapfilled_[measured].to_numpy(), self.df.loc[measured, 'NEE'].to_numpy())

    def test_lloyd_taylor_recovers_the_respiration(self):
        p = self.wn.wetnight_params_
        self.assertEqual(list(p.index), [2020, 2021, 2022])
        self.assertTrue(((p['E0'] - E0).abs() < 0.15 * E0).all(), p)
        filled = self.wn.get_flag() == FLAG_WETNIGHT
        truth = lloyd_taylor(self.ta[filled].to_numpy(), RREF, E0)
        self.assertLess(np.abs(self.wn.gapfilled_[filled].to_numpy() - truth).mean(), 0.3)

    def test_year_with_too_few_dry_nights_keeps_xgboost(self):
        df, night, wet, ta = self.df, self.night, self.wet, self.ta
        dry = ~wet & (df.index.year != 2021)  # no dry nights in 2021
        wn = LongTermGapFillingXGBoostWetNightTS(input_df=df, target_col='NEE', wet=wet, ta=ta, night=night,
                                                 dry=dry, **XGB).run()
        self.assertNotIn(2021, wn.wetnight_params_.index)
        in2021 = wn.gapfilled_.index.year == 2021
        self.assertFalse((wn.get_flag()[in2021] == FLAG_WETNIGHT).any())
        np.testing.assert_allclose(wn.gapfilled_[in2021].to_numpy(), self.plain.gapfilled_[in2021].to_numpy())

    def test_rejects_non_series(self):
        with self.assertRaises(TypeError):
            LongTermGapFillingXGBoostWetNightTS(input_df=self.df, target_col='NEE', wet=self.wet.to_numpy(),
                                                ta=self.ta, night=self.night, **XGB)


class TestRunLevel41XgbWetNight(unittest.TestCase):

    def test_chain_runs_next_to_xgboost(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.core.ml.feature_engineer import FeatureEngineer
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector, run_level32,
            run_level33_constant_ustar, run_level41_xgb, run_level41_xgb_wetnight)
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)
        df['WET'] = (df.index.dayofyear % 7) == 0  # one wet day per week
        data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(data, thresholds=[0.1], threshold_labels=['CUT_50'],
                                          showplot=False, verbose=False)
        features = ['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa']
        engineer = FeatureEngineer(target_col='_target_', features_lag=[-1, 1], vectorize_timestamps=True, verbose=0)
        data = run_level41_xgb(data, features=features, engineer=engineer, verbose=0, **XGB)
        data = run_level41_xgb_wetnight(data, features=features, engineer=engineer, wet='WET', ta='TA_1_1_1',
                                        min_records=10, verbose=0, **XGB)
        cols = data.gapfilled_cols()
        self.assertEqual(set(cols), {'xgb', 'xgbwn'})
        xg, wn = data.fpc_df[cols['xgb']['CUT_50']], data.fpc_df[cols['xgbwn']['CUT_50']]
        self.assertTrue(cols['xgbwn']['CUT_50'].endswith('_gfXGWN'))
        flag = data.fpc_df[f"FLAG_{cols['xgbwn']['CUT_50']}_ISFILLED"]
        lt = flag == FLAG_WETNIGHT
        self.assertGreater(int(lt.sum()), 0)
        self.assertTrue((data.fpc_df.loc[lt, 'NIGHTTIME'] == 1).all())
        self.assertTrue(df.loc[lt[lt].index, 'WET'].all())
        np.testing.assert_allclose(wn[~lt].to_numpy(), xg[~lt].to_numpy())
        self.assertIn('xgbwn', data.levels.level41_methods())


class TestRunChainWetNight(unittest.TestCase):
    """FluxConfig.gapfill_xgb_wetnight in run_chain."""

    @classmethod
    def setUpClass(cls):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import init_flux_data
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)
        df['WET'] = (df.index.dayofyear % 7) == 0  # one wet day per week
        cls.df = df
        cls.data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)

    def _config(self, **kw):
        from diive.flux.postprocessingchain import FluxConfig
        base = dict(fluxcol='FC', ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
                    level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
                    gapfill_rf=False, gapfill_xgb=False, gapfill_mds=False,
                    gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa'],
                    gapfill_reduce_features=False)
        base.update(kw)
        return FluxConfig(**base)

    def test_validation(self):
        from diive.flux.postprocessingchain import run_chain
        with self.assertRaisesRegex(ValueError, 'wetnight_wet'):
            run_chain(self.data, self._config(gapfill_xgb_wetnight=True, wetnight_ta='TA_1_1_1'))
        with self.assertRaisesRegex(ValueError, 'wetnight_ta or mds_ta'):
            run_chain(self.data, self._config(gapfill_xgb_wetnight=True, wetnight_wet='WET'))
        with self.assertRaisesRegex(KeyError, "wetnight_wet='NOT_THERE'"):
            run_chain(self.data, self._config(gapfill_xgb_wetnight=True, wetnight_wet='NOT_THERE',
                                              wetnight_ta='TA_1_1_1'))
        with self.assertRaisesRegex(ValueError, 'gapfill_xgb_wetnight=True'):
            run_chain(self.data, self._config(partition_nighttime_oneflux=True, partition_gapfill_method='xgbwn',
                                              partition_ta='TA_1_1_1', partition_sw_in='SW_IN_1_1_1',
                                              partition_ta_f='TA_1_1_1'))

    def test_run_chain_fills_the_wet_nights(self):
        from diive.flux.postprocessingchain import run_chain
        # TA falls back to mds_ta, which is set although MDS is off.
        out = run_chain(self.data, self._config(gapfill_xgb_wetnight=True, wetnight_wet='WET',
                                                mds_ta='TA_1_1_1', wetnight_min_records=10))
        self.assertEqual(set(out.levels.level41_methods()), {'xgbwn'})
        col = out.gapfilled_cols()['xgbwn']['CUT_50']
        self.assertTrue(col.endswith('_gfXGWN'))
        inst = out.levels.level41_xgbwn['CUT_50']
        self.assertEqual(inst.min_records, 10)
        lt = out.fpc_df[f'FLAG_{col}_ISFILLED'] == FLAG_WETNIGHT
        self.assertGreater(int(lt.sum()), 0)
        self.assertTrue(self.df.loc[lt[lt].index, 'WET'].all())


if __name__ == '__main__':
    unittest.main()
