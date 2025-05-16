import unittest

import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
from diive.configs.exampledata import load_exampledata_parquet
from diive.pkgs.createvar.timesince import TimeSince


class TestCreateVar(unittest.TestCase):

    def test_conversion_et_from_le(self):
        """Calculate ET from LE and compare results to ET calculated by EddyPro."""
        df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        le = df['LE'].copy()
        et_eddypro = df['ET'].copy()  # Should be in mm h-1
        ta = df['TA_1_1_1'].copy()
        et = dv.et_from_le(le=le, ta=ta)
        self.assertAlmostEqual(et[0], et_eddypro[0], places=4)
        self.assertAlmostEqual(et[1], et_eddypro[1], places=4)
        self.assertAlmostEqual(et[-1], et_eddypro[-1], places=3)
        self.assertAlmostEqual(et.sum(), et_eddypro.sum(), places=0)

    def test_lagged_variants(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.createvar.laggedvariants import lagged_variants
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
        from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag
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
        self.assertEqual(results.sum().sum(), 52180821.63268461)
        self.assertEqual(swin_pot.sum(), 52005525.63268461)
        self.assertEqual(daytime_flag.sum(), 87592)
        self.assertEqual(daytime_flag.max(), 1)
        self.assertEqual(daytime_flag.min(), 0)
        self.assertEqual(nighttime_flag.sum(), 87704)
        self.assertEqual(nighttime_flag.max(), 1)
        self.assertEqual(nighttime_flag.min(), 0)
        self.assertEqual(daytime_flag[nighttime_flag == 0].min(), 1)
        self.assertEqual(daytime_flag[nighttime_flag == 0].max(), 1)
        self.assertEqual(nighttime_flag[daytime_flag == 0].min(), 1)
        self.assertEqual(nighttime_flag[daytime_flag == 0].max(), 1)

    def test_calc_vpd(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.createvar.vpd import calc_vpd_from_ta_rh  # Used to calculate VPD
        ta_col = 'Tair_f'  # Air temperature (gap-filled) is used to calculate VPD
        rh_col = 'RH'  # Relative humidity (not gap-filled) is used to calculate VPD
        vpd_col = 'VPD_hPa'  # VPD will be newly calculated from gap-filled TA and non-gap-filled RH
        df = load_exampledata_parquet()
        subsetcols = [ta_col, rh_col]
        subset_df = df[subsetcols].copy()
        subset_df[vpd_col] = calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)
        self.assertEqual(subset_df[vpd_col].sum(), 56371.50662138253)
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


if __name__ == '__main__':
    unittest.main()
