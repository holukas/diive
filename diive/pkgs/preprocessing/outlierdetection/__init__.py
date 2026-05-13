"""
OUTLIER DETECTION: MULTI-METHOD FLAGGING
=========================================

Statistical and density-based methods for identifying anomalies: Hampel, z-score, local SD,
Local Outlier Factor, trim, absolute limits, manual removal, and incremental change detection.
Chain multiple methods sequentially via StepwiseOutlierDetection.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.pkgs.preprocessing.outlierdetection.absolutelimits import (
    AbsoluteLimits,
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
    zScoreRolling,
)

# Aliases for compatibility
AbsoluteLimitsDaytimeNighttime = AbsoluteLimits
HampelDaytimeNighttime = Hampel
hampel = Hampel
hampel_daytime_nighttime = Hampel
absolute_limits_daytime_nighttime = AbsoluteLimitsDaytimeNighttime
zscore_increments = zScoreIncrements
zscore = zScore
LocalOutlierFactorAllData = LocalOutlierFactor
LocalOutlierFactorDaytimeNighttime = LocalOutlierFactor
zscore_rolling = zScoreRolling

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
    'zScoreRolling',
    'zscore_rolling',
]
