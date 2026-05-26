"""
OUTLIER DETECTION: MULTI-METHOD FLAGGING
=========================================

Statistical and density-based methods for identifying anomalies: Hampel, z-score, local SD,
Local Outlier Factor, trim, absolute limits, manual removal, and incremental change detection.
Chain multiple methods sequentially via StepwiseOutlierDetection.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.preprocessing.outlier_detection.absolutelimits import (
    AbsoluteLimits,
)
from diive.preprocessing.outlier_detection.hampel import Hampel
from diive.preprocessing.outlier_detection.incremental import zScoreIncrements
from diive.preprocessing.outlier_detection.localsd import LocalSD
from diive.preprocessing.outlier_detection.lof import LocalOutlierFactor
from diive.preprocessing.outlier_detection.manualremoval import ManualRemoval
from diive.preprocessing.outlier_detection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.preprocessing.outlier_detection.trim import TrimLow
from diive.preprocessing.outlier_detection.zscore import (
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
