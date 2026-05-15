# TODO generalization bias
# TODO SHAP values
# https://pypi.org/project/shap/
# https://mljar.com/blog/feature-importance-in-random-forest/

"""
GAP-FILLING: XGBOOST TIME SERIES IMPUTATION
============================================

Gradient boosting gap-filling with XGBoost.
Feature engineering pipeline and model scoring included.

Part of the diive library: https://github.com/holukas/diive
"""
import xgboost as xgb
from pandas import DataFrame

from diive.core.ml.common import MlRegressorGapFillingBase


class XGBoostTS(MlRegressorGapFillingBase):

    def __init__(self, input_df: DataFrame, target_col: str or tuple, verbose: int = 0,
                 test_size: float = 0.25, **kwargs):
        """Gap-filling for time series using XGBoost gradient boosting.

        Trains an XGBoost model on complete observations to predict missing values.
        Effective for non-linear relationships and complex temporal patterns.
        Requires pre-engineered features from FeatureEngineer.

        Args:
            input_df: DataFrame with target and pre-engineered feature columns.
                     Features should be created with FeatureEngineer.
                     Timestamps in DatetimeIndex.
            target_col: Column name of variable to gap-fill (string or tuple).
            verbose: Verbosity level: 0=silent, 1=progress, 2+=detailed.
                    Default: 0.
            test_size: Fraction of complete data for testing (0.0-1.0).
                      Default: 0.25. Only complete rows used for split.
            **kwargs: XGBoost hyperparameters (n_estimators, max_depth,
                     learning_rate, min_child_weight, early_stopping_rounds,
                     random_state, n_jobs, subsample, colsample_bytree, etc).
                     See xgboost documentation.

        Methods:
            trainmodel(): Train on training data, evaluate on test data.
            fillgaps(): Train on all complete data, predict missing values.
            reduce_features(): Feature selection based on SHAP importance.
            report_traintest(): Print model evaluation metrics.
            report_gapfilling(): Print gap-filling results.
            get_gapfilled_target(): Return gap-filled series.
            get_flag(): Return gap-filling flags (0=observed, 1=gap-filled, 2=fallback).

        Attributes:
            model_: Trained XGBRegressor instance.
            gapfilling_df_: DataFrame with gap-filled target.
            feature_importances_: SHAP feature importance.
            scores_: Model performance metrics (MAE, RMSE, R²).

        Examples:
            See examples/pkgs/gapfilling/gapfill_xgboost_ts.py for complete example.
            See examples/pkgs/gapfilling/gapfill_comparison.py for side-by-side comparison
            with MDS and Random Forest gap-filling.
        """

        # Pass to parent class
        super().__init__(
            regressor=xgb.XGBRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            test_size=test_size,
            **kwargs
        )


