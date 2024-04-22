import unittest

import pandas as pd

import diive.configs.exampledata as ed
from diive.pkgs.createvar.noise import add_impulse_noise
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits


# kudos https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

class TestOutlierDetection(unittest.TestCase):

    def test_absolute_limits(self):
        """Load EddyPro _fluxnet_ file"""
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
