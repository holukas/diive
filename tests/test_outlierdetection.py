import unittest

import numpy as np
import pandas as pd

import diive.configs.exampledata as ed
from diive.variables import add_impulse_noise
from diive.preprocessing.outlier_detection import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.preprocessing.outlier_detection.hampel import HampelDaytimeNighttime
from diive.preprocessing.outlier_detection import zScoreIncrements
from diive.preprocessing.outlier_detection import LocalSD
from diive.preprocessing.outlier_detection import LocalOutlierFactorAllData
from diive.preprocessing.outlier_detection import TrimLow
from diive.preprocessing.outlier_detection import zScore, zScoreRolling
from diive.preprocessing.qaqc import MissingValues


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

        # Counts encode the potrad day/night split (ONEFlux/FLUXNET parity), which
        # classifies these 1488 July records; the twilight edges decide a handful of
        # them, and two land on the day side, so trim_nighttime does not trim them.
        # Checks on bad data
        badmean = checkdf.loc[checkdf.flag == 2, 's_noise'].mean()
        self.assertEqual(badmean, 19.914995180241654)
        baddata_stats = checkdf.loc[checkdf.flag == 2].describe()
        self.assertEqual(baddata_stats.loc['max']['s_noise'], 338.9234661966423)
        self.assertEqual(baddata_stats.loc['min']['s_noise'], -12.230067031003944)
        self.assertEqual(baddata_stats.loc['count']['flag'], 376)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 376)

        # Checks on good data
        goodmean = checkdf.loc[checkdf.flag == 0, 's_noise'].mean()
        self.assertEqual(goodmean, 17.82873128107376)
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 338.3652327597214)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -40.33549755756406)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1112)

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
            n_sigma_daytime=5.5,
            n_sigma_nighttime=5.5,
            window_length=48 * 3,
            use_differencing=False,
            separate_day_night=True,
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
        self.assertEqual(baddata_stats.loc['count']['flag'], 92)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 92)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 33.72896422933141)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -11.769722879346313)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1396)

    def test_hampel_filter_daytime_nighttime_doublediff(self):
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
            n_sigma_daytime=100,
            n_sigma_nighttime=100,
            window_length=48,
            use_differencing=True,
            separate_day_night=True,
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
        self.assertEqual(baddata_stats.loc['count']['flag'], 227)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 227)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 53.93064794948049)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -24.68608044400628)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1261)

    def test_hampel_filter_basic(self):
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
            n_sigma_daytime=5.5,
            n_sigma_nighttime=5.5,
            window_length=48,
            use_differencing=False,
            separate_day_night=False,
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
        self.assertEqual(baddata_stats.loc['count']['flag'], 92)
        self.assertEqual(baddata_stats.loc['min']['flag'], 2)
        self.assertEqual(baddata_stats.loc['max']['flag'], 2)
        self.assertEqual(baddata_stats.loc['count']['s_noise'], 92)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertEqual(gooddata_stats.loc['max']['s_noise'], 51.98160999845608)
        self.assertEqual(gooddata_stats.loc['min']['s_noise'], -7.383053125047635)
        self.assertEqual(gooddata_stats.loc['min']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['max']['flag'], 0)
        self.assertEqual(gooddata_stats.loc['count']['s_noise'], 1396)

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

        zdn = zScore(
            series=s_noise,
            separate_day_night=True,
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

    def test_localsd_daytime_nighttime(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-11,
                                    factor_high=9,
                                    contamination=0.2,
                                    seed=42)  # Add impulse noise (spikes)
        lsd = LocalSD(
            series=s_noise,
            separate_day_night=True,
            n_sd_daytime=3,
            n_sd_nighttime=2,
            winsize_daytime=48 * 2,
            winsize_nighttime=48 * 1,
            constant_sd=False,
            lat=46.0,
            lon=11.0,
            utc_offset=1,
            showplot=False,
            verbose=False
        )
        lsd.calc(repeat=True)
        flag = lsd.get_flag()
        frame = {'s_noise': s_noise, 'flag': flag}
        checkdf = pd.DataFrame.from_dict(frame)

        # Get daytime and nighttime data
        checkdf['FLAG_DAYTIME'] = lsd.flag_daytime
        checkdf['FLAG_NIGHTTIME'] = lsd.flag_nighttime
        good_dt = checkdf.loc[(checkdf['FLAG_DAYTIME'] == 1) & (checkdf['flag'] == 0)].copy()
        bad_dt = checkdf.loc[(checkdf['FLAG_DAYTIME'] == 1) & (checkdf['flag'] == 2)].copy()
        good_nt = checkdf.loc[(checkdf['FLAG_NIGHTTIME'] == 1) & (checkdf['flag'] == 0)].copy()
        bad_nt = checkdf.loc[(checkdf['FLAG_NIGHTTIME'] == 1) & (checkdf['flag'] == 2)].copy()

        # Counts encode the potrad day/night split (ONEFlux/FLUXNET parity); the
        # twilight edges decide which set a record lands in, and the four counts
        # total 1488.
        # Checks on good data
        good_dt_stats = good_dt.describe()
        self.assertEqual(good_dt_stats.loc['max']['s_noise'], 31.87658379873561)
        self.assertEqual(good_dt_stats.loc['min']['s_noise'], 4.3238046734031155)
        self.assertEqual(good_dt_stats.loc['count']['s_noise'], 766)
        good_nt_stats = good_nt.describe()
        self.assertEqual(good_nt_stats.loc['max']['s_noise'], 17.073)
        self.assertEqual(good_nt_stats.loc['min']['s_noise'], 5.262)
        self.assertEqual(good_nt_stats.loc['count']['s_noise'], 376)

        # Checks on bad data
        bad_dt_stats = bad_dt.describe()
        self.assertEqual(bad_dt_stats.loc['max']['s_noise'], 231.78475439289213)
        self.assertEqual(bad_dt_stats.loc['min']['s_noise'], -38.52634400343396)
        self.assertEqual(bad_dt_stats.loc['count']['s_noise'], 168)
        bad_nt_stats = bad_nt.describe()
        self.assertEqual(bad_nt_stats.loc['max']['s_noise'], 224.8390630344748)
        self.assertEqual(bad_nt_stats.loc['min']['s_noise'], -36.84146979488223)
        self.assertEqual(bad_nt_stats.loc['count']['s_noise'], 178)

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

    def test_localsd_values_exactly_on_the_limit_are_ok(self):
        # `ok` used strict comparisons while `rejected` did too, so a value
        # sitting exactly on a limit was in neither set. A constant series is
        # the float-exact case: sd = 0 puts both limits right on the data, so
        # every single record hit it and none was reported as ok.
        s = pd.Series(data=5.0, name='CONSTANT',
                      index=pd.date_range('2022-06-01', periods=96, freq='30min'))
        lsd = LocalSD(series=s, n_sd=4, winsize=10, showplot=False, verbose=False)
        ok, rejected, n_outliers, upper, lower = lsd._identify_outliers(
            s=s, winsize=10, n_sd=4, iteration=1)
        # The limits really are on the data, i.e. this is the boundary case.
        self.assertEqual(float(upper.iloc[10]), 5.0)
        self.assertEqual(float(lower.iloc[10]), 5.0)
        # ok and rejected must partition the series, no record in neither.
        self.assertEqual(len(rejected), 0)
        self.assertEqual(n_outliers, 0)
        self.assertEqual(len(ok), len(s))

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
            minval_daytime=daytime_minmax[0],
            maxval_daytime=daytime_minmax[1],
            minval_nighttime=nighttime_minmax[0],
            maxval_nighttime=nighttime_minmax[1],
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


class TestVerboseStatistics(unittest.TestCase):
    """verbose=True must print the per-iteration statistics the docstrings promise.
    They go through detail(), which prints from VERBOSE_DEBUG, while verbose=True
    maps to VERBOSE_PROGRESS - so the statistics used to be unreachable."""

    def _series(self):
        idx = pd.date_range('2020-05-01', periods=500, freq='1min', name='TIMESTAMP_MIDDLE')
        rng = np.random.default_rng(3)
        s = pd.Series(20 + rng.normal(0, 0.05, len(idx)), index=idx, name='SWC')
        s.iloc[250] = 40.0
        return s

    def _run(self, verbose):
        from contextlib import redirect_stdout
        from io import StringIO
        from diive.preprocessing.outlier_detection.hampel import Hampel

        buf = StringIO()
        with redirect_stdout(buf):
            ham = Hampel(series=self._series(), lat=47.478333, lon=8.364389, utc_offset=1,
                         window_length=60, n_sigma=8, use_differencing=True,
                         separate_day_night=False, showplot=False, verbose=verbose)
            ham.calc(repeat=False)
        return buf.getvalue()

    def test_verbose_true_prints_outlier_counts(self):
        self.assertIn('Outliers', self._run(verbose=True))

    def test_verbose_false_stays_quiet(self):
        self.assertNotIn('Outliers', self._run(verbose=False))


class TestDaytimeNighttimeNames(unittest.TestCase):
    """The *DaytimeNighttime names must do what they say.

    They used to be plain aliases for their base class, so
    LocalOutlierFactorDaytimeNighttime was the same object as
    LocalOutlierFactorAllData -- two names meaning opposite things -- and
    both ran on the whole series.
    """

    @staticmethod
    def _diel_series(periods: int = 48 * 120):
        idx = pd.date_range('2024-06-01', periods=periods, freq='30min')
        hours = idx.hour + idx.minute / 60
        rng = np.random.default_rng(0)
        return pd.Series(12 * np.sin((hours - 6) / 24 * 2 * np.pi) + 8
                         + rng.normal(0, 1.5, periods), index=idx, name='TA')

    COORDS = dict(lat=46.815333, lon=9.855972, utc_offset=1)

    def test_lof_daytime_nighttime_is_not_alldata(self):
        from diive.preprocessing.outlier_detection import (
            LocalOutlierFactorAllData, LocalOutlierFactorDaytimeNighttime)
        self.assertIsNot(LocalOutlierFactorDaytimeNighttime, LocalOutlierFactorAllData)

        s = self._diel_series()

        def n_flagged(cls):
            d = cls(series=s.copy(), n_neighbors=20, contamination=0.01, **self.COORDS)
            d.calc(repeat=False)
            return int((d.overall_flag == 2).sum())

        # Separating changes the neighbourhoods, so the two must disagree.
        self.assertNotEqual(n_flagged(LocalOutlierFactorDaytimeNighttime),
                            n_flagged(LocalOutlierFactorAllData))

    def test_absolutelimits_daytime_nighttime_applies_per_period_limits(self):
        s = self._diel_series(periods=48 * 10)

        # Per-period overrides are the intended usage.
        al = AbsoluteLimitsDaytimeNighttime(series=s.copy(),
                                            minval_daytime=4.0, maxval_daytime=25.0,
                                            minval_nighttime=-5.0, maxval_nighttime=10.0,
                                            **self.COORDS)
        al.calc(repeat=False)
        self.assertGreater(int((al.overall_flag == 2).sum()), 0)

        # minval/maxval alone cover both periods, per the shared day/night
        # convention. AbsoluteLimits is pointwise, so equal limits on both sides
        # must give exactly the same flags as not separating at all.
        split = AbsoluteLimitsDaytimeNighttime(series=s.copy(), minval=-5, maxval=25, **self.COORDS)
        split.calc(repeat=False)
        whole = AbsoluteLimits(series=s.copy(), minval=-5, maxval=25, separate_day_night=False)
        whole.calc(repeat=False)
        self.assertEqual(int((split.overall_flag == 2).sum()),
                         int((whole.overall_flag == 2).sum()))

        # The removed pair name reports its replacement.
        with self.assertRaises(TypeError) as ctx:
            AbsoluteLimitsDaytimeNighttime(series=s.copy(), daytime_minmax=[4.0, 25.0],
                                           nighttime_minmax=[-5.0, 10.0], **self.COORDS)
        self.assertIn('minval_daytime', str(ctx.exception))


class TestRenamedParamsRejected(unittest.TestCase):
    """Removed parameter names must name their replacement.

    Detectors take **legacy purely so a pre-unification call gets a useful
    message. If that regressed, the old name would land in **kwargs and be
    ignored, silently running with the wrong settings.
    """

    @staticmethod
    def _series(n: int = 200):
        idx = pd.date_range('2024-06-01', periods=n, freq='30min')
        return pd.Series(np.arange(float(n)), index=idx, name='TA')

    COORDS = dict(lat=46.8, lon=9.8, utc_offset=1)

    def test_old_switch_name_names_its_replacement(self):
        from diive.preprocessing.outlier_detection import LocalSD, LocalOutlierFactor
        for cls, extra in ((zScore, {}), (LocalSD, {}), (LocalOutlierFactor, {}),
                           (AbsoluteLimits, dict(minval=0, maxval=1))):
            with self.subTest(cls=getattr(cls, '__name__', str(cls))):
                with self.assertRaises(TypeError) as ctx:
                    cls(series=self._series(), separate_daytime_nighttime=True,
                        **self.COORDS, **extra)
                msg = str(ctx.exception)
                self.assertIn('separate_daytime_nighttime', msg)
                self.assertIn('separate_day_night', msg)

    def test_old_hampel_short_names_rejected(self):
        with self.assertRaises(TypeError) as ctx:
            HampelDaytimeNighttime(series=self._series(), n_sigma_dt=5, **self.COORDS)
        self.assertIn('n_sigma_daytime', str(ctx.exception))

    def test_localsd_rejects_the_old_list_form(self):
        from diive.preprocessing.outlier_detection import LocalSD
        # n_sd kept its name and only stopped accepting a list, so no name-based
        # check can catch this; it needs its own type guard.
        with self.assertRaises(TypeError) as ctx:
            LocalSD(series=self._series(), n_sd=[4, 7], separate_day_night=True, **self.COORDS)
        msg = str(ctx.exception)
        self.assertIn('n_sd_daytime', msg)
        self.assertIn('n_sd_nighttime', msg)

    def test_a_plain_typo_still_raises_normally(self):
        # **legacy must not become a silent catch-all for misspellings.
        with self.assertRaises(TypeError) as ctx:
            zScore(series=self._series(), thres_zscoer=4)
        self.assertIn('unexpected keyword argument', str(ctx.exception))
