import unittest

import numpy as np
import pandas as pd

import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
from diive.configs.exampledata import load_exampledata_FLUXNET_FULLSET_HH_CSV_30MIN
from diive.configs.exampledata import load_exampledata_parquet
from diive.variables import air_temp_from_sonic_temp
from diive.variables import TimeSince


class TestCreateVar(unittest.TestCase):

    def test_air_temp_from_sonic_temp(self):
        # Sonic temperature in Kelvin
        sonic_temp = pd.Series([287.549, 287.540, 287.552, 287.556, 287.559,
                                287.566, 287.560, 287.562, 287.557, 287.560],
                               name='sonic_temp')
        # H2O in mol mol-1
        h2o = pd.Series([0.013417, 0.013453, 0.013492, 0.013419, 0.013476,
                         0.013503, 0.013463, 0.013472, 0.013521, 0.013481],
                        name='h2o')
        # Pre-computed expected result
        expected_air_temp = pd.Series([286.319673, 286.307446, 286.315829, 286.326464, 286.324287,
                                       286.328795, 286.326439, 286.327595, 286.318199, 286.324774],
                                      name='TA_SONIC')
        air_temp = air_temp_from_sonic_temp(sonic_temp=sonic_temp, h2o=h2o)
        pd.testing.assert_series_equal(expected_air_temp, air_temp)

    def test_conversion_et_from_le(self):
        """Calculate ET from LE and compare results to ET calculated by EddyPro."""
        df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        le = df['LE'].copy()
        et_eddypro = df['ET'].copy()  # Should be in mm h-1
        ta = df['TA_1_1_1'].copy()
        et = dv.variables.et_from_le(le=le, ta=ta)
        self.assertAlmostEqual(et.iloc[0], et_eddypro.iloc[0], places=4)
        self.assertAlmostEqual(et.iloc[1], et_eddypro.iloc[1], places=4)
        self.assertAlmostEqual(et.iloc[-1], et_eddypro.iloc[-1], places=3)
        self.assertAlmostEqual(et.sum(), et_eddypro.sum(), places=0)

    def test_lagged_variants(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.variables import lagged_variants
        df = load_exampledata_parquet()
        df = load_exampledata_parquet()
        locs = (df.index.year == 2022) & (df.index.month == 7) & (df.index.hour >= 10) & (df.index.hour <= 15)
        df = df[locs].copy()
        df = df[['Tair_f', 'Rg_f', 'NEE_CUT_REF_f']].copy()
        results = lagged_variants(
            df=df,
            lag=[-2, 1],
            stepsize=1,
            exclude_cols=['NEE_CUT_REF_f'],  # Variable(s) that will not be lagged
            verbose=True
        )
        self.assertEqual(results.sum().sum(), 1109117.4049999998)
        self.assertEqual(len(results.columns), 9)
        self.assertEqual(results.columns.to_list(),
                         ['Tair_f', 'Rg_f', 'NEE_CUT_REF_f', '.Tair_f-2', '.Tair_f-1', '.Tair_f+1', '.Rg_f-2',
                          '.Rg_f-1', '.Rg_f+1'])

        self.assertEqual(list(results['Tair_f'].iloc[0:4]), [8.04, 7.94, 8.15, 7.85])
        self.assertEqual(list(results['.Tair_f-2'].iloc[0:4]), [8.04, 8.04, 8.04, 7.94])
        self.assertEqual(list(results['.Tair_f-1'].iloc[0:4]), [8.04, 8.04, 7.94, 8.15])
        self.assertEqual(list(results['.Tair_f+1'].iloc[0:4]), [7.94, 8.15, 7.85, 7.69])

    def test_daytime_nighttime_flag(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.variables import DaytimeNighttimeFlag
        df = load_exampledata_parquet()
        dnf = DaytimeNighttimeFlag(
            timestamp_index=df.index,
            nighttime_threshold=1,
            lat=47.286417,
            lon=7.733750,
            utc_offset=1
        )
        results = dnf.get_results()
        swin_pot = dnf.get_swinpot()
        daytime_flag = dnf.get_daytime_flag()
        nighttime_flag = dnf.get_nighttime_flag()
        # Baselines encode potrad (ONEFlux/FLUXNET parity): solar constant 1376,
        # Spencer 1971 declination/eccentricity, NOAA solar-noon shift, period mean.
        self.assertAlmostEqual(results.sum().sum(), 52742196.78324184, places=3)
        self.assertAlmostEqual(swin_pot.sum(), 52566900.78324184, places=3)
        self.assertEqual(daytime_flag.sum(), 90888)
        self.assertEqual(daytime_flag.max(), 1)
        self.assertEqual(daytime_flag.min(), 0)
        self.assertEqual(nighttime_flag.sum(), 84408)
        self.assertEqual(nighttime_flag.max(), 1)
        self.assertEqual(nighttime_flag.min(), 0)
        self.assertEqual(daytime_flag[nighttime_flag == 0].min(), 1)
        self.assertEqual(daytime_flag[nighttime_flag == 0].max(), 1)
        self.assertEqual(nighttime_flag[daytime_flag == 0].min(), 1)
        self.assertEqual(nighttime_flag[daytime_flag == 0].max(), 1)

    def test_calc_vpd(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.variables import calc_vpd_from_ta_rh  # Used to calculate VPD
        ta_col = 'Tair_f'  # Air temperature (gap-filled) is used to calculate VPD
        rh_col = 'RH'  # Relative humidity (not gap-filled) is used to calculate VPD
        vpd_col = 'VPD_hPa'  # VPD will be newly calculated from gap-filled TA and non-gap-filled RH
        df = load_exampledata_parquet()
        subsetcols = [ta_col, rh_col]
        subset_df = df[subsetcols].copy()
        subset_df[vpd_col] = calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)
        self.assertAlmostEqual(subset_df[vpd_col].sum(), 56371.50662138253, places=3)
        self.assertEqual(subset_df[vpd_col].min(), 0)
        self.assertEqual(subset_df[vpd_col].max(), 3.215734681690522)
        self.assertEqual(subset_df[vpd_col].dropna().count(), 174589)

    def test_timesince(self):
        df = load_exampledata_parquet()
        series_ta = df.loc[(df.index.year == 2022) & (df.index.month == 3), "Tair_f"].copy()
        ts = TimeSince(series_ta, upper_lim=5, lower_lim=None, include_lim=True)
        ts.calc()
        ts_full_results = ts.get_full_results()
        greater_equal_stats = ts_full_results.loc[ts_full_results['Tair_f'] >= 5].describe()
        less_stats = ts_full_results.loc[ts_full_results['Tair_f'] < 5].describe()
        self.assertEqual(greater_equal_stats['Tair_f']['count'], 273)
        self.assertEqual(greater_equal_stats['Tair_f']['min'], 5.017)
        self.assertEqual(ts_full_results['FLAG_IS_OUTSIDE_RANGE'].sum(), 273)
        self.assertEqual(less_stats['Tair_f']['count'], 1215)
        self.assertEqual(less_stats['Tair_f']['max'], 4.99)
        self.assertEqual(less_stats['FLAG_IS_OUTSIDE_RANGE']['min'], 0)
        self.assertEqual(less_stats['FLAG_IS_OUTSIDE_RANGE']['max'], 0)
        self.assertEqual(ts_full_results.sum().sum(), -7223.621999999999)
        # from pathlib import Path
        # outpath = Path(r"F:\TMP") / 'ts_full_results.csv'
        # ts_full_results.to_csv(outpath, index=False)
        # ts_series = ts.get_timesince()

    def test_aerodynamic_resistance(self):
        from diive.variables import aerodynamic_resistance
        u = pd.Series([2.0, 4.0, 1.0, 3.0], name='u')
        ustar = pd.Series([0.5, 0.4, 0.0, 0.5], name='ustar')  # ustar=0 -> NaN
        ra = aerodynamic_resistance(u_ms=u, ustar_ms=ustar)
        # ra = u / ustar^2
        self.assertAlmostEqual(ra.iloc[0], 2.0 / 0.5 ** 2, places=6)
        self.assertAlmostEqual(ra.iloc[1], 4.0 / 0.4 ** 2, places=6)
        self.assertTrue(np.isnan(ra.iloc[2]))  # ustar <= 0 -> NaN

    def test_dry_air_density(self):
        from diive.variables import dry_air_density
        rho_a = pd.Series([1.20, 1.18, 1.22], name='rho_a')
        rho_v = pd.Series([0.01, 0.012, 0.009], name='rho_v')
        rho_d = dry_air_density(rho_a=rho_a, rho_v=rho_v)
        self.assertTrue(np.allclose(rho_d.to_numpy(), (rho_a - rho_v).to_numpy()))

    def test_latent_heat_of_vaporization(self):
        from diive.variables import latent_heat_of_vaporization
        ta = pd.Series([0.0, 20.0, 30.0], name='ta')
        lv = latent_heat_of_vaporization(ta=ta)
        # (2.501 - 0.00237*ta) * 1e6
        self.assertAlmostEqual(lv.iloc[0], 2.501e6, places=0)
        self.assertAlmostEqual(lv.iloc[1], (2.501 - 0.00237 * 20.0) * 1e6, places=0)
        # decreases with temperature, stays in a physical range (~2.4-2.5 MJ/kg)
        self.assertTrue(lv.iloc[2] < lv.iloc[1] < lv.iloc[0])
        self.assertTrue((lv > 2.4e6).all() and (lv < 2.55e6).all())

    def test_potrad_fluxnet_parity(self):
        # Ground truth: real SW_IN_POT column from a FLUXNET2015 FULLSET file (CH-Cha),
        # produced by the actual ONEFlux/FLUXNET pipeline. TIMESTAMP is TIMESTAMP_MIDDLE
        # after loading (diive convention).
        from diive.variables import potrad
        df, _ = load_exampledata_FLUXNET_FULLSET_HH_CSV_30MIN()
        lat, lon, utc_offset = 47.210227, 8.410645, 1
        truth = df['SW_IN_POT']
        swin_pot = potrad(timestamp_index=df.index, lat=lat, lon=lon, utc_offset=utc_offset)
        self.assertEqual(swin_pot.name, 'SW_IN_POT')
        maxdiff = (swin_pot.to_numpy() - truth.to_numpy())
        maxdiff = np.abs(maxdiff).max()
        self.assertLess(maxdiff, 2)  # W m-2, real pipeline peak is ~480 W m-2

    def test_potrad_physical_sanity(self):
        from diive.variables import potrad
        # Full year, mid-latitude northern site, correct MIDDLE timestamps for 30min data.
        idx = pd.date_range('2022-01-01 00:15', '2022-12-31 23:45', freq='30min')
        swin_pot = potrad(timestamp_index=idx, lat=47.0, lon=8.0, utc_offset=1)
        self.assertFalse(swin_pot.isna().any())
        self.assertTrue((swin_pot >= 0).all())  # never negative
        self.assertEqual(swin_pot[swin_pot.index.hour == 0].max(), 0)  # night is zero
        # Annual peak of a northern-hemisphere site falls near the summer solstice (~doy 172).
        peak_doy = swin_pot.idxmax().dayofyear
        self.assertLess(abs(peak_doy - 172), 15)

    def test_potrad_southern_hemisphere(self):
        from diive.variables import potrad
        idx = pd.date_range('2022-01-01 00:15', '2022-12-31 23:45', freq='30min')
        north = potrad(timestamp_index=idx, lat=47.0, lon=8.0, utc_offset=1)
        south = potrad(timestamp_index=idx, lat=-47.0, lon=8.0, utc_offset=1)
        # Southern-hemisphere peak falls near the December solstice (~doy 355).
        south_peak_doy = south.idxmax().dayofyear
        self.assertLess(min(abs(south_peak_doy - 355), abs(south_peak_doy - 355 + 365)), 15)
        # Earth-sun eccentricity: closest to the sun in January, so the southern peak
        # (northern winter) is higher than the comparable northern peak (northern summer).
        self.assertGreater(south.max(), north.max())

    def test_potrad_resolutions(self):
        from diive.variables import potrad
        # Hourly
        idx_hourly = pd.date_range('2022-06-20 00:30', '2022-06-22 23:30', freq='1h')
        r_hourly = potrad(timestamp_index=idx_hourly, lat=47.0, lon=8.0, utc_offset=1)
        self.assertEqual(len(r_hourly), len(idx_hourly))
        self.assertFalse(r_hourly.isna().any())
        # 30-min
        idx_30min = pd.date_range('2022-06-20 00:15', '2022-06-22 23:45', freq='30min')
        r_30min = potrad(timestamp_index=idx_30min, lat=47.0, lon=8.0, utc_offset=1)
        self.assertEqual(len(r_30min), len(idx_30min))
        self.assertFalse(r_30min.isna().any())
        # Leap year (spans the Feb 29, 2008 leap day)
        idx_leap = pd.date_range('2008-02-27 00:30', '2008-03-01 23:30', freq='1h')
        r_leap = potrad(timestamp_index=idx_leap, lat=47.0, lon=8.0, utc_offset=1)
        self.assertEqual(len(r_leap), len(idx_leap))
        self.assertFalse(r_leap.isna().any())

    def test_potrad_errors(self):
        from diive.variables import potrad
        # A single timestamp cannot yield an inferred averaging period.
        idx_single = pd.date_range('2022-06-21 12:00', periods=1, freq='30min')
        with self.assertRaises(ValueError):
            potrad(timestamp_index=idx_single, lat=47.0, lon=8.0, utc_offset=1)

    def test_potrad_odd_frequencies(self):
        """Periods that do not tile the day, sub-minute records and windows
        crossing New Year all resolve instead of raising."""
        from diive.variables import potrad
        for freq in ['7min', '13min', '30s']:
            idx = pd.date_range('2022-06-21 00:00', '2022-06-21 23:59', freq=freq)
            swinpot = potrad(timestamp_index=idx, lat=47.0, lon=8.0, utc_offset=1)
            self.assertEqual(len(swinpot), len(idx))
            self.assertFalse(swinpot.isna().any())
            self.assertGreater(swinpot.max(), 1000)  # midsummer peak is resolved
            self.assertGreaterEqual(swinpot.min(), 0)

        # A window straddling the New Year boundary uses each side's own year.
        idx_ny = pd.date_range('2022-12-31 20:00', '2023-01-01 04:00', freq='11min')
        swinpot_ny = potrad(timestamp_index=idx_ny, lat=47.0, lon=8.0, utc_offset=1)
        self.assertFalse(swinpot_ny.isna().any())

        # An irregular index (a gap) is unaffected: the period is the median spacing.
        idx_full = pd.date_range('2022-06-01 00:15', '2022-06-30 23:45', freq='30min')
        idx_gappy = idx_full.delete(np.arange(100, 500))
        gappy = potrad(timestamp_index=idx_gappy, lat=47.0, lon=8.0, utc_offset=1)
        full = potrad(timestamp_index=idx_full, lat=47.0, lon=8.0, utc_offset=1)
        np.testing.assert_allclose(gappy.to_numpy(), full.reindex(idx_gappy).to_numpy())

    def test_potrad_odd_period_windows_march_uniformly(self):
        """An odd averaging period puts window starts on half-minutes, which the
        1-minute grid cannot represent. Rounding those halves to even (np.rint)
        would make consecutive windows step 6, 8, 6, 8 ... minutes apart for a
        7-minute period, so the curve climbs in alternating jumps instead of
        evenly. Rounding half up keeps the march uniform."""
        from diive.variables import potrad
        # Sunrise ramp: the steepest part of the curve, where uneven window
        # spacing shows up most strongly.
        idx = pd.date_range('2018-06-21 04:00', '2018-06-21 09:00', freq='7min')
        swinpot = potrad(timestamp_index=idx, lat=47.286417, lon=7.733750, utc_offset=1)

        steps = np.diff(swinpot.to_numpy())
        steep = steps[steps > 15]  # the roughly linear stretch of the ramp
        self.assertGreater(len(steep), 20)  # guard: the ramp was actually found
        # Consecutive steps up a near-linear ramp should be near-equal. Rounding
        # halves to even gives ~0.32 here; rounding half up gives ~0.02.
        unevenness = np.abs(np.diff(steep)).max() / np.median(steep)
        self.assertLess(unevenness, 0.1)


if __name__ == '__main__':
    unittest.main()


class TestLaggedVariantsSingleColumn(unittest.TestCase):
    """A one-column dataframe used to come back with no lags and no warning."""

    @staticmethod
    def _df(cols):
        idx = pd.date_range('2024-01-01', periods=10, freq='30min', name='TIMESTAMP_MIDDLE')
        return pd.DataFrame({c: np.arange(10.0) for c in cols}, index=idx)

    def test_single_column_is_lagged(self):
        from diive.variables import lagged_variants
        out = lagged_variants(df=self._df(['TA']), lag=[-2, 1], verbose=0)
        self.assertEqual(list(out.columns), ['TA', '.TA-2', '.TA-1', '.TA+1'])

    def test_single_column_matches_the_same_column_among_others(self):
        from diive.variables import lagged_variants
        # Excluding the second column leaves exactly one column to lag, which is
        # the case that always worked. A lone column must behave identically.
        alone = lagged_variants(df=self._df(['TA']), lag=[-2, 1], verbose=0)
        among = lagged_variants(df=self._df(['TA', 'X']), lag=[-2, 1],
                                exclude_cols=['X'], verbose=0)
        lagcols = ['.TA-2', '.TA-1', '.TA+1']
        self.assertEqual([c for c in among.columns if c.startswith('.')], lagcols)
        for c in lagcols:
            self.assertTrue(alone[c].equals(among[c]), f'{c} differs')

    def test_single_excluded_column_still_raises(self):
        from diive.variables import lagged_variants
        # Nothing left to lag: this is the case the guard was written for.
        with self.assertRaises(Exception):
            lagged_variants(df=self._df(['TA']), lag=[-2, 1], exclude_cols=['TA'], verbose=0)


class TestClassifyVariable(unittest.TestCase):
    """`classify_variable` decides every variable pill colour in the GUI, and is
    the library's authoritative "what kind of column is this"."""

    def test_recognised_names_map_to_kind_and_category(self):
        from diive.variables import classify_variable
        from diive.variables.classification import (
            CATEGORY_CARBON, CATEGORY_METEO, CATEGORY_NITROGEN,
            CATEGORY_RADIATION, CATEGORY_SOIL, CATEGORY_WATER)
        cases = {
            'NEE_CUT_REF_f': ('NEE', CATEGORY_CARBON),
            'GPP_DT_CUT_REF': ('GPP', CATEGORY_CARBON),
            'Reco_NT_CUT_REF': ('Reco', CATEGORY_CARBON),
            'FCH4': ('FCH4', CATEGORY_CARBON),
            'FN2O_1_1_1': ('FN2O', CATEGORY_NITROGEN),
            'FH2O': ('FH2O', CATEGORY_WATER),
            'LE_f': ('LE', CATEGORY_WATER),
            'ET_1_1_1': ('ET', CATEGORY_WATER),
            'Rg_f': ('Rg', CATEGORY_RADIATION),
            'SW_IN_T1_2_1': ('SW_IN', CATEGORY_RADIATION),
            'PPFD_IN': ('PPFD', CATEGORY_RADIATION),
            'PAR_1_1_1': ('PAR', CATEGORY_RADIATION),
            'LW_IN_1_1_1': ('LW', CATEGORY_RADIATION),
            'Tair_f': ('TA', CATEGORY_METEO),
            'TA_T1_2_1': ('TA', CATEGORY_METEO),
            'VPD_f': ('VPD', CATEGORY_METEO),
            'SWC_GF1_0.05_1': ('SWC', CATEGORY_SOIL),
        }
        for name, (kind, category) in cases.items():
            with self.subTest(name=name):
                result = classify_variable(name)
                self.assertIsNotNone(result, f"{name} should be classified")
                self.assertEqual((result.kind, result.category), (kind, category))

    def test_fc_does_not_swallow_fch4(self):
        from diive.variables import classify_variable
        # FC is the CO2 flux; a plain "FC" prefix rule would also catch FCH4
        # (methane), so FC is matched on a word boundary.
        self.assertEqual(classify_variable('FC').kind, 'FC')
        self.assertEqual(classify_variable('FC_1_1_1').kind, 'FC')
        self.assertEqual(classify_variable('FCH4').kind, 'FCH4')
        self.assertEqual(classify_variable('FCH4_1_1_1').kind, 'FCH4')

    def test_bare_ta_is_exact_matched(self):
        from diive.variables import classify_variable
        # A "TA" prefix rule would also catch TARGET / TAU, so bare TA is exact.
        self.assertEqual(classify_variable('TA').kind, 'TA')
        self.assertIsNone(classify_variable('TARGET'))
        self.assertIsNone(classify_variable('TAU'))

    def test_unrecognised_names_return_none(self):
        from diive.variables import classify_variable
        for name in ('H', 'USTAR', 'RANDOM_COLUMN', ''):
            with self.subTest(name=name):
                self.assertIsNone(classify_variable(name))
        # Non-string input is tolerated rather than raising.
        self.assertIsNone(classify_variable(None))
        self.assertIsNone(classify_variable(3.14))

    def test_result_is_a_named_tuple(self):
        from diive.variables import VariableClass, classify_variable
        result = classify_variable('NEE_CUT_REF_f')
        self.assertIsInstance(result, VariableClass)
        # Usable both by field and by position.
        kind, category = result
        self.assertEqual((kind, category), (result.kind, result.category))


class TestAutoPickColumn(unittest.TestCase):
    """Name-based seeding for the GUI's variable pickers."""

    COLUMNS = ['TIMESTAMP', 'TA_T1_2_1', 'SW_IN_T1_2_1', 'SW_IN_POT',
               'SW_OUT_T1_2_1', 'VPD_f', 'NEE_CUT_REF_f']

    def test_first_match_wins(self):
        from diive.variables import auto_pick_column
        self.assertEqual(auto_pick_column(self.COLUMNS, 'SW_IN'), 'SW_IN_T1_2_1')
        self.assertEqual(auto_pick_column(self.COLUMNS, 'VPD'), 'VPD_f')

    def test_prefer_ranks_a_match_first(self):
        from diive.variables import auto_pick_column
        # Without prefer, the plain SW_IN column wins by position; with it, the
        # POT column is promoted.
        self.assertEqual(auto_pick_column(self.COLUMNS, 'SW_IN', prefer='POT'),
                         'SW_IN_POT')

    def test_avoid_excludes_columns(self):
        from diive.variables import auto_pick_column
        self.assertEqual(auto_pick_column(self.COLUMNS, 'SW_IN', avoid='POT'),
                         'SW_IN_T1_2_1')
        # avoid also applies to the preferred pass.
        self.assertEqual(
            auto_pick_column(self.COLUMNS, 'SW_IN', prefer='POT', avoid='POT'),
            'SW_IN_T1_2_1')

    def test_no_match_returns_empty_string(self):
        from diive.variables import auto_pick_column
        self.assertEqual(auto_pick_column(self.COLUMNS, 'CH4'), '')
        self.assertEqual(auto_pick_column([], 'TA'), '')

    def test_matching_is_against_the_uppercased_column(self):
        from diive.variables import auto_pick_column
        # The needle is compared to the upper-cased name, so a lower-case column
        # still matches an upper-case needle.
        self.assertEqual(auto_pick_column(['tair_f'], 'TAIR'), 'tair_f')


class TestCombineVariables(unittest.TestCase):
    """`combine_variables` backs the GUI's Combine-variables tab."""

    def setUp(self):
        self.idx = pd.date_range('2021-01-01', periods=4, freq='30min',
                                 name='TIMESTAMP_MIDDLE')
        self.a = pd.Series([1.0, 2.0, np.nan, 4.0], index=self.idx, name='A')
        self.b = pd.Series([10.0, np.nan, 30.0, 40.0], index=self.idx, name='B')

    def test_arithmetic_methods(self):
        from diive.variables import combine_variables
        expected = {
            'add': [11.0, np.nan, np.nan, 44.0],
            'subtract': [-9.0, np.nan, np.nan, -36.0],
            'multiply': [10.0, np.nan, np.nan, 160.0],
            'divide': [0.1, np.nan, np.nan, 0.1],
        }
        for method, values in expected.items():
            with self.subTest(method=method):
                out = combine_variables(self.a, self.b, method=method)
                np.testing.assert_allclose(out.to_numpy(), values)

    def test_keep_overlap_only_false_uses_the_operation_identity(self):
        from diive.variables import combine_variables
        # A missing operand becomes the identity, so a one-sided record survives:
        # 0 for add/subtract, 1 for multiply/divide.
        add = combine_variables(self.a, self.b, method='add', keep_overlap_only=False)
        np.testing.assert_allclose(add.to_numpy(), [11.0, 2.0, 30.0, 44.0])
        mul = combine_variables(self.a, self.b, method='multiply', keep_overlap_only=False)
        np.testing.assert_allclose(mul.to_numpy(), [10.0, 2.0, 30.0, 160.0])
        sub = combine_variables(self.a, self.b, method='subtract', keep_overlap_only=False)
        np.testing.assert_allclose(sub.to_numpy(), [-9.0, 2.0, -30.0, -36.0])

    def test_fillgaps_keeps_series1_and_fills_only_its_gaps(self):
        from diive.variables import combine_variables
        out = combine_variables(self.a, self.b, method='fillgaps')
        # Position 2 is A's gap -> takes B's 30.0; every other record keeps A.
        np.testing.assert_allclose(out.to_numpy(), [1.0, 2.0, 30.0, 4.0])

    def test_fillgaps_ignores_keep_overlap_only(self):
        from diive.variables import combine_variables
        # Filling gaps is a union by definition, so the flag must not change it.
        a = combine_variables(self.a, self.b, method='fillgaps', keep_overlap_only=True)
        b = combine_variables(self.a, self.b, method='fillgaps', keep_overlap_only=False)
        pd.testing.assert_series_equal(a, b)

    def test_result_index_is_the_union(self):
        from diive.variables import combine_variables
        later = pd.Series([5.0], name='B',
                          index=pd.date_range('2021-01-02', periods=1, freq='30min',
                                              name='TIMESTAMP_MIDDLE'))
        out = combine_variables(self.a, later, method='add', keep_overlap_only=False)
        self.assertEqual(len(out), len(self.a) + 1)

    def test_default_and_custom_names(self):
        from diive.variables import combine_variables
        self.assertEqual(combine_variables(self.a, self.b, method='add').name,
                         'A_ADD_B')
        self.assertEqual(
            combine_variables(self.a, self.b, method='add', name='MY_SUM').name,
            'MY_SUM')

    def test_unknown_method_raises_and_lists_the_choices(self):
        from diive.variables import combine_variables
        with self.assertRaises(ValueError) as ctx:
            combine_variables(self.a, self.b, method='exponentiate')
        message = str(ctx.exception)
        self.assertIn('exponentiate', message)
        self.assertIn('fillgaps', message)


class TestDaytimeNighttimeFlagFromSwinpot(unittest.TestCase):
    """The standalone flag function (the `DaytimeNighttimeFlag` class is covered
    by TestCreateVar.test_daytime_nighttime_flag)."""

    def setUp(self):
        self.idx = pd.date_range('2021-06-01', periods=5, freq='30min',
                                 name='TIMESTAMP_MIDDLE')
        self.swinpot = pd.Series([0.0, 10.0, 20.0, 100.0, 19.999],
                                 index=self.idx, name='SW_IN_POT')

    def test_threshold_is_inclusive_for_daytime(self):
        from diive.variables import daytime_nighttime_flag_from_swinpot
        daytime, nighttime = daytime_nighttime_flag_from_swinpot(
            self.swinpot, nighttime_threshold=20)
        # >= threshold is daytime, so exactly 20.0 counts as day.
        np.testing.assert_allclose(daytime.to_numpy(), [0, 0, 1, 1, 0])
        np.testing.assert_allclose(nighttime.to_numpy(), [1, 1, 0, 0, 1])

    def test_flags_are_complementary(self):
        from diive.variables import daytime_nighttime_flag_from_swinpot
        daytime, nighttime = daytime_nighttime_flag_from_swinpot(self.swinpot)
        np.testing.assert_allclose((daytime + nighttime).to_numpy(), 1.0)

    def test_threshold_changes_the_split(self):
        from diive.variables import daytime_nighttime_flag_from_swinpot
        low, _ = daytime_nighttime_flag_from_swinpot(self.swinpot, nighttime_threshold=5)
        high, _ = daytime_nighttime_flag_from_swinpot(self.swinpot, nighttime_threshold=50)
        self.assertEqual(int(low.sum()), 4)
        self.assertEqual(int(high.sum()), 1)

    def test_output_column_names_are_configurable(self):
        from diive.variables import daytime_nighttime_flag_from_swinpot
        daytime, nighttime = daytime_nighttime_flag_from_swinpot(
            self.swinpot, daytime_col='IS_DAY', nighttime_col='IS_NIGHT')
        self.assertEqual(daytime.name, 'IS_DAY')
        self.assertEqual(nighttime.name, 'IS_NIGHT')
        self.assertTrue(daytime.index.equals(self.swinpot.index))
