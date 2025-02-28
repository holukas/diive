import unittest

import numpy as np
import pandas as pd

import diive.configs.exampledata as ed
from diive.pkgs.createvar.noise import add_impulse_noise
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.outlierdetection.hampel import Hampel, HampelDaytimeNighttime
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.localsd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.trim import TrimLow
from diive.pkgs.outlierdetection.zscore import zScore, zScoreDaytimeNighttime, zScoreRolling
from diive.pkgs.qaqc.flags import MissingValues


# kudos https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

class TestOutlierDetection(unittest.TestCase):

    def test_zscore_rolling(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-15,
                                    factor_high=14,
                                    contamination=0.03,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        zsr = zScoreRolling(
            series=s_noise,
            thres_zscore=4,
            winsize=50,
            showplot=True,
            verbose=False
        )
        zsr.calc()
        flag = zsr.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        badmean = checkdf.loc[checkdf.flag == 2, 's_noise'].mean()
        self.assertEqual(badmean, 176.562204534145)
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 338.9234661966423)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -40.33549755756406)
        self.assertEqual(baddata_stats.loc['count']['flag'], 40)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 40)

        # Checks on good data
        goodmean = checkdf.loc[checkdf.flag == 0, 's_noise'].mean()
        self.assertEqual(goodmean, 13.98556573961263)
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 28.472316256494835)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 2.151442210229117)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1448)

    def test_missing_values(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        # Delete some data points
        s.iloc[500:600] = np.nan
        s.iloc[721:791] = np.nan
        mv = MissingValues(series=s)
        mv.calc()
        flag = mv.get_flag()
        n_missing_vals = int(flag.loc[flag == 2].count())
        n_available_vals = int(flag.loc[flag == 0].count())
        n_total_vals = n_available_vals + n_missing_vals
        self.assertEqual(n_missing_vals, int(s.isnull().sum()))
        self.assertEqual(n_available_vals, int(s.count()))
        self.assertEqual(n_total_vals, len(s))

    def test_trim_low_nt(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-15,
                                    factor_high=14,
                                    contamination=0.03,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        trm = TrimLow(
            series=s_noise,
            trim_daytime=False,
            trim_nighttime=True,
            lower_limit=10,
            showplot=False,
            verbose=False,
            lat=47.286417,
            lon=7.733750,
            utc_offset=1
        )
        trm.calc()
        flag = trm.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        badmean = checkdf.loc[checkdf.flag == 2, 's_noise'].mean()
        self.assertEqual(badmean, 19.926195230975953)
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 338.9234661966423)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -12.230067031003944)
        self.assertEqual(baddata_stats.loc['count']['flag'], 380)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 380)

        # Checks on good data
        goodmean = checkdf.loc[checkdf.flag == 0, 's_noise'].mean()
        self.assertEqual(goodmean, 17.8173584698141)
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 338.3652327597214)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -40.33549755756406)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1108)

    def test_hampel_filter_daytime_nighttime(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-12,
                                    factor_high=17,
                                    contamination=0.07,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        ham = HampelDaytimeNighttime(
            series=s_noise,
            n_sigma_dt=4,
            n_sigma_nt=3,
            window_length=48 * 9,
            showplot=False,
            verbose=False,
            lat=47.286417,
            lon=7.733750,
            utc_offset=1
        )
        ham.calc(repeat=True)
        flag = ham.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 420.37816376334473)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -38.04507418841196)
        self.assertEqual(baddata_stats.loc['count']['flag'], 102)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 102)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 30.473344762824773)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 5.049)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1386)

    def test_hampel_filter(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-14,
                                    factor_high=19,
                                    contamination=0.06,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        ham = Hampel(
            series=s_noise,
            n_sigma=4,
            window_length=48 * 9,
            showplot=False,
            verbose=False
        )
        ham.calc(repeat=True)
        flag = ham.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 457.7993015844542)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -47.71027579882934)
        self.assertEqual(baddata_stats.loc['count']['flag'], 82)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 82)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 25.201551539241972)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 3.65795487099882)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1406)

    def test_zscore(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-15,
                                    factor_high=26,
                                    contamination=0.04,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        zsc = zScore(
            series=s_noise,
            thres_zscore=4,
            showplot=False,
            verbose=False)

        zsc.calc(repeat=True)
        flag = zsc.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 623.9300725355847)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -51.46751403512717)
        self.assertEqual(baddata_stats.loc['count']['flag'], 57)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 57)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 25.723642479636727)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 1.187508723671586)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1431)

    def test_zscore_daytime_nighttime(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-19,
                                    factor_high=6,
                                    contamination=0.02,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        zdn = zScoreDaytimeNighttime(
            series=s_noise,
            lat=47.286417,
            lon=7.733750,
            utc_offset=1,
            thres_zscore=4,
            showplot=False,
            verbose=False)

        zdn.calc(repeat=True)
        flag = zdn.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 148.72806841344465)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -68.17770769831958)
        self.assertEqual(baddata_stats.loc['count']['flag'], 26)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 26)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 27.376145041037773)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 2.810267874163495)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1462)

    def test_lof_alldata(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-10,
                                    factor_high=3,
                                    contamination=0.04,
                                    seed=42)  # Add impulse noise (spikes)
        lofa = LocalOutlierFactorAllData(
            series=s_noise,
            n_neighbors=1200,
            contamination='auto',
            showplot=False,
            n_jobs=-1
        )
        lofa.calc(repeat=True)
        flag = lofa.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 79.16756136930726)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -30.530597715816295)
        self.assertEqual(baddata_stats.loc['count']['flag'], 47)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 47)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 24.344)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 2.8838640536716156)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1441)

    def test_localsd_with_constantsd(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-11,
                                    factor_high=9,
                                    contamination=0.2,
                                    seed=42)  # Add impulse noise (spikes)
        lsd = LocalSD(series=s_noise,
                      n_sd=2,
                      winsize=48 * 10,
                      constant_sd=True,
                      showplot=False,
                      verbose=False)
        lsd.calc(repeat=True)
        flag = lsd.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 231.78475439289213)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -38.52634400343396)
        self.assertEqual(baddata_stats.loc['count']['flag'], 715)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 715)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 16.276)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 6.315)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 773)

    def test_localsd(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-10,
                                    factor_high=3,
                                    contamination=0.04,
                                    seed=42)  # Add impulse noise (spikes)
        lsd = LocalSD(series=s_noise,
                      n_sd=4,
                      winsize=48 * 10,
                      constant_sd=False,
                      showplot=False,
                      verbose=False)
        lsd.calc(repeat=True)
        flag = lsd.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 79.16756136930726)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -30.530597715816295)
        self.assertEqual(baddata_stats.loc['count']['flag'], 44)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 44)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 31.43947292041035)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -2.1888477232075214)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1444)

    def test_zscore_increments(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-20,
                                    factor_high=5,
                                    contamination=0.04,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        zsi = zScoreIncrements(series=s_noise,
                               thres_zscore=4.5,
                               showplot=False,
                               verbose=False)

        zsi.calc(repeat=True)
        flag = zsi.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 124.94945003274493)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -80.29042252645108)
        self.assertEqual(baddata_stats.loc['count']['flag'], 56)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 56)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 24.344)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 5.049)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1432)

    def test_absolute_limits(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-20,
                                    factor_high=5,
                                    contamination=0.04,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        al = AbsoluteLimits(series=s_noise, minval=10, maxval=22)
        al.calc()
        flag = al.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Checks on bad data
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 124.94945003274493)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -80.29042252645108)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], 10)
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 22)

    def test_absolute_limits_dt_nt(self):
        """Load EddyPro _fluxnet_ file"""
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-20,
                                    factor_high=5,
                                    contamination=0.08,
                                    seed=42)  # Add impulse noise (spikes)

        # Checks on noise data, make sure we have outliers, i.e., greater or less than the specified limits
        self.assertGreater(s_noise.max(), 22)
        self.assertLess(s_noise.min(), 10)

        daytime_minmax = [4.0, 25.0]
        nighttime_minmax = [-5.0, 10.0]
        al = AbsoluteLimitsDaytimeNighttime(
            series=s_noise,
            lat=46.815333,
            lon=9.855972,
            utc_offset=1,
            daytime_minmax=daytime_minmax,
            nighttime_minmax=nighttime_minmax
        )
        al.calc(repeat=False)
        flag = al.get_flag()
        frame = {'s': s, 's_noise': s_noise, 'flag': flag, 'is_daytime': al.is_daytime.astype(int),
                 'is_nighttime': al.is_nighttime.astype(int)}
        checkdf = pd.DataFrame.from_dict(frame)

        nt_min_s = checkdf.loc[checkdf['is_nighttime'] == 1]['s'].min()
        nt_max_s = checkdf.loc[checkdf['is_nighttime'] == 1]['s'].max()
        dt_min_s = checkdf.loc[checkdf['is_daytime'] == 1]['s'].min()
        dt_max_s = checkdf.loc[checkdf['is_daytime'] == 1]['s'].max()

        nt_min_s_noise = checkdf.loc[checkdf['is_nighttime'] == 1]['s_noise'].min()
        nt_max_s_noise = checkdf.loc[checkdf['is_nighttime'] == 1]['s_noise'].max()
        dt_min_s_noise = checkdf.loc[checkdf['is_daytime'] == 1]['s_noise'].min()
        dt_max_s_noise = checkdf.loc[checkdf['is_daytime'] == 1]['s_noise'].max()

        # Check if we have indeed spike outliers, required for next assertions
        self.assertLess(nt_min_s_noise, nt_min_s)
        self.assertLess(dt_min_s_noise, dt_min_s)
        self.assertGreater(nt_max_s_noise, nt_max_s)
        self.assertGreater(dt_max_s_noise, dt_max_s)

        # Collect good daytime data and make sure their min and max values are within the limits
        gooddata_dt = checkdf.loc[(checkdf['flag'] == 0) & (checkdf['is_daytime'] == 1)].copy()
        gooddata_dt_stats = gooddata_dt.describe()
        self.assertGreaterEqual(gooddata_dt_stats.loc['min']['s_noise'], daytime_minmax[0])
        self.assertLessEqual(gooddata_dt_stats.loc['max']['s_noise'], daytime_minmax[1])

        # Collect good nighttime data and make sure their min and max values are within the limits
        gooddata_nt = checkdf.loc[(checkdf['flag'] == 0) & (checkdf['is_nighttime'] == 1)].copy()
        gooddata_nt_stats = gooddata_nt.describe()
        self.assertGreaterEqual(gooddata_nt_stats.loc['min']['s_noise'], nighttime_minmax[0])
        self.assertLessEqual(gooddata_nt_stats.loc['max']['s_noise'], nighttime_minmax[1])
