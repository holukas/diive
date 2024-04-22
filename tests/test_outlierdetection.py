import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.pkgs.createvar.noise import add_impulse_noise
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag

# kudos https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

class TestOutlierDetection(unittest.TestCase):

    def test_absolute_limits(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-20,
                                    factor_high=5,
                                    contamination=0.04)  # Add impulse noise (spikes)

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
        self.assertGreater(baddata_stats.loc['max']['s_noise'], 22)
        self.assertLess(baddata_stats.loc['min']['s_noise'], 10)

        # Checks on good data
        gooddata_stats = checkdf.loc[checkdf.flag == 0].describe()
        self.assertGreaterEqual(gooddata_stats.loc['min']['s_noise'], 10)
        self.assertLessEqual(gooddata_stats.loc['max']['s_noise'], 22)

    def test_absolute_limits_dt_nt(self):
        """Load EddyPro _fluxnet_ file"""
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        s = s.loc[s.index.month == 7].copy()
        s_noise = add_impulse_noise(series=s,
                                    factor_low=-20,
                                    factor_high=5,
                                    contamination=0.08)  # Add impulse noise (spikes)

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
        frame = {'s': s, 's_noise': s_noise, 'flag': flag, 'is_daytime': al.is_daytime.astype(int), 'is_nighttime': al.is_nighttime.astype(int)}
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
