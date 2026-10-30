"""
OUTLIERS: OUTLIER DETECTION
===========================

Public namespace ``dv.outliers``: single outlier detection methods.
For several tests in sequence, use ``StepwiseOutlierDetection`` from
``diive.preprocessing.outlier_detection``.

Part of the diive library: https://github.com/holukas/diive
"""
from diive.preprocessing.outlier_detection import AbsoluteLimits
from diive.preprocessing.outlier_detection import AbsoluteLimitsDaytimeNighttime
from diive.preprocessing.outlier_detection import LocalOutlierFactor
from diive.preprocessing.outlier_detection import LocalOutlierFactorAllData
from diive.preprocessing.outlier_detection import LocalOutlierFactorDaytimeNighttime
from diive.preprocessing.outlier_detection import LocalSD
from diive.preprocessing.outlier_detection import ManualRemoval
from diive.preprocessing.outlier_detection import NegativeNighttime
from diive.preprocessing.outlier_detection import NegativeNights
from diive.preprocessing.outlier_detection import TrimLow
from diive.preprocessing.outlier_detection import zScore
from diive.preprocessing.outlier_detection import zScoreIncrements
from diive.preprocessing.outlier_detection import zScoreRolling
from diive.preprocessing.outlier_detection.hampel import Hampel
from diive.preprocessing.outlier_detection.hampel import HampelDaytimeNighttime

__all__ = [
    'AbsoluteLimits',
    'AbsoluteLimitsDaytimeNighttime',
    'LocalOutlierFactor',
    'LocalOutlierFactorAllData',
    'LocalOutlierFactorDaytimeNighttime',
    'LocalSD',
    'ManualRemoval',
    'NegativeNighttime',
    'NegativeNights',
    'TrimLow',
    'zScore',
    'zScoreIncrements',
    'zScoreRolling',
    'Hampel',
    'HampelDaytimeNighttime',
]
