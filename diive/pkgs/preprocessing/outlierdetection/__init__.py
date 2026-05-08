from diive.pkgs.preprocessing.outlierdetection.absolutelimits import (
    AbsoluteLimits,
    AbsoluteLimitsDaytimeNighttime,
)
from diive.pkgs.preprocessing.outlierdetection.hampel import Hampel
from diive.pkgs.preprocessing.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.preprocessing.outlierdetection.localsd import LocalSD
from diive.pkgs.preprocessing.outlierdetection.lof import LocalOutlierFactor
from diive.pkgs.preprocessing.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.preprocessing.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.preprocessing.outlierdetection.trim import TrimLow
from diive.pkgs.preprocessing.outlierdetection.zscore import (
    zScore,
    zScoreDaytimeNighttime,
    zScoreRolling,
)

# Aliases for compatibility
HampelDaytimeNighttime = Hampel
hampel = Hampel
hampel_daytime_nighttime = Hampel
absolute_limits_daytime_nighttime = AbsoluteLimitsDaytimeNighttime
zscore_increments = zScoreIncrements
zscore = zScore
LocalOutlierFactorAllData = LocalOutlierFactor
LocalOutlierFactorDaytimeNighttime = LocalOutlierFactor

__all__ = [
    'AbsoluteLimits',
    'AbsoluteLimitsDaytimeNighttime',
    'absolute_limits_daytime_nighttime',
    'Hampel',
    'HampelDaytimeNighttime',
    'hampel',
    'hampel_daytime_nighttime',
    'zScoreIncrements',
    'zscore_increments',
    'LocalSD',
    'LocalOutlierFactor',
    'LocalOutlierFactorAllData',
    'LocalOutlierFactorDaytimeNighttime',
    'ManualRemoval',
    'StepwiseOutlierDetection',
    'TrimLow',
    'zScore',
    'zscore',
    'zScoreDaytimeNighttime',
    'zScoreRolling',
]
