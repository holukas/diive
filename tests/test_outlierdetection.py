import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.pkgs.createvar.noise import add_impulse_noise
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.localsd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.zscore import zScore, zScoreDaytimeNighttime
from diive.pkgs.outlierdetection.hampel import Hampel


# kudos https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

class TestOutlierDetection(unittest.TestCase):

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
            showplot=True,
            verbose=True
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
        al.calc(repeat=False)
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
