"""
GAP-FILLING: TIME SERIES IMPUTATION METHODS
============================================

ML and statistical methods for filling missing data: Random Forest, XGBoost, MDS, linear interpolation.
Feature engineering pipeline and model scoring included.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.core.ml.feature_engineer import FeatureEngineer
from diive.core.ml.results import GapFillingResult
from diive.core.ml.optimization import OptimizeParamsTS
from diive.gapfilling.interpolate import linear_interpolation
from diive.gapfilling.mds import FluxMDS
from diive.gapfilling.randomforest_ts import RandomForestTS, QuickFillRFTS, OptimizeParamsRFTS
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.gapfilling.scores import prediction_scores
from diive.gapfilling.longterm import LongTermGapFillingRandomForestTS, LongTermGapFillingXGBoostTS
from diive.gapfilling.swin import SWINGapFillerXGBoost

__all__ = [
    'FeatureEngineer',
    'GapFillingResult',
    'linear_interpolation',
    'FluxMDS',
    'RandomForestTS',
    'QuickFillRFTS',
    'OptimizeParamsTS',
    'OptimizeParamsRFTS',
    'XGBoostTS',
    'prediction_scores',
    'LongTermGapFillingRandomForestTS',
    'LongTermGapFillingXGBoostTS',
    'SWINGapFillerXGBoost',
]
