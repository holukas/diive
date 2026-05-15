# TODO generalization bias
# TODO SHAP values
# https://pypi.org/project/shap/
# https://mljar.com/blog/feature-importance-in-random-forest/

"""
GAP-FILLING: RANDOM FOREST TIME SERIES IMPUTATION
=================================================

Ensemble learning gap-filling with scikit-learn Random Forest.
Feature engineering pipeline and model scoring included.

Part of the diive library: https://github.com/holukas/diive
"""
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor  # Import the model we are using

import diive.core.dfun.frames as fr
from diive.core.ml.common import MlRegressorGapFillingBase
from diive.core.ml.optimization import OptimizeParamsTS

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


# Backward compatibility alias for existing code
OptimizeParamsRFTS = OptimizeParamsTS


class RandomForestTS(MlRegressorGapFillingBase):

    def __init__(self,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: bool = True,
                 test_size: float = 0.25,
                 **kwargs):
        """Gap-filling for time series using Random Forest regression.

        Trains a Random Forest model on complete observations to predict missing values.
        Combines robustness to outliers with interpretability through feature importance
        analysis. Requires pre-engineered features from FeatureEngineer.

        Args:
            input_df: DataFrame with target and pre-engineered feature columns.
                     Features should be created with FeatureEngineer.
                     Timestamps in DatetimeIndex.
            target_col: Column name of variable to gap-fill (string or tuple).
            verbose: Verbosity level: 0=silent, 1=progress, 2+=detailed.
                    Default: True (equivalent to 1).
            test_size: Fraction of complete data for testing (0.0-1.0).
                      Default: 0.25. Only complete rows used for split.
            **kwargs: Random Forest hyperparameters (n_estimators, max_depth,
                     min_samples_split, min_samples_leaf, random_state, n_jobs, etc).
                     See scikit-learn RandomForestRegressor documentation.

        Methods:
            trainmodel(): Train on training data, evaluate on test data.
            fillgaps(): Train on all complete data, predict missing values.
            reduce_features(): Feature selection based on SHAP importance.
            report_traintest(): Print model evaluation metrics.
            report_gapfilling(): Print gap-filling results.
            get_gapfilled_target(): Return gap-filled series.
            get_flag(): Return gap-filling flags (0=observed, 1=gap-filled, 2=fallback).

        Attributes:
            model_: Trained RandomForestRegressor instance.
            gapfilling_df_: DataFrame with gap-filled target.
            feature_importances_: SHAP feature importance.
            scores_: Model performance metrics (MAE, RMSE, R²).

        Examples:
            See examples/pkgs/gapfilling/gapfill_randomforest_ts.py for complete examples.
            See examples/pkgs/gapfilling/gapfill_comparison.py for side-by-side comparison
            with MDS and XGBoost gap-filling.
        """

        # Pass to parent class
        super().__init__(
            regressor=RandomForestRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            test_size=test_size,
            **kwargs
        )


class QuickFillRFTS:
    """
    Quick gap-filling using RandomForestTS with pre-defined minimal parameters

    The purpose of this class is preliminary/exploratory gap-filling for quick tests
    and rapid prototyping. It is NOT meant for production/final gap-filling.

    Uses minimal feature engineering and model complexity for speed:
    - FeatureEngineer with single lag: [-1, -1] (only immediate past value)
    - No rolling statistics, differencing, or timestamp features
    - RandomForestTS with n_estimators=3 (very fast, low quality)
    - Shallow trees with larger min_samples for fast inference

    Attributes:
        gapfilling_df(): DataFrame with gap-filled target and flags
        get_gapfilled_target(): Series with gap-filled values
        get_flag(): Series with gap-filling flags (0=observed, 1=filled, 2=fallback)
    """

    def __init__(self, df: DataFrame, target_col: str or tuple):
        from diive.core.ml.feature_engineer import FeatureEngineer

        self.df = df.copy()
        self.target_col = target_col
        self.rfts = None

        # Step 1: Engineer minimal features for fast gap-filling
        engineer = FeatureEngineer(
            target_col=self.target_col,
            features_lag=[-1, -1],  # Only immediate past value
            features_lag_stepsize=1,
            features_lag_exclude_cols=None,
            features_rolling=None,  # No rolling stats (speed)
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,  # No differencing (speed)
            features_diff_exclude_cols=None,
            features_ema=None,  # No EMA (speed)
            features_ema_exclude_cols=None,
            features_poly_degree=None,  # No polynomial (speed)
            features_poly_exclude_cols=None,
            features_stl=False,  # No STL (speed)
            features_stl_method='stl',
            features_stl_seasonal_period=None,
            features_stl_exclude_cols=None,
            features_stl_components=None,
            vectorize_timestamps=True,
            add_continuous_record_number=False,  # No extra features (speed)
            sanitize_timestamp=False,  # Skip validation (speed)
        )
        df_engineered = engineer.fit_transform(self.df)

        # Step 2: Create gap-filling model with engineered features
        self.rfts = RandomForestTS(
            input_df=df_engineered,
            target_col=self.target_col,
            verbose=True,
            n_estimators=9,  # Minimal trees for speed
            random_state=42,
            min_samples_split=10,  # Large minimum for fast trees
            min_samples_leaf=5,  # Large minimum for fast trees
            max_depth=None,
            n_jobs=-1
        )

    def fill(self):
        self.rfts.trainmodel(showplot_scores=False, showplot_importance=False)
        self.rfts.fillgaps(showplot_scores=False, showplot_importance=False)

    def gapfilling_df(self):
        return self.rfts.gapfilling_df_

    def report(self):
        return self.rfts.report_gapfilling()

    def get_gapfilled_target(self) -> Series:
        return self.rfts.get_gapfilled_target()

    def get_flag(self) -> Series:
        return self.rfts.get_flag()
