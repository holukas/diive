# TODO generalization bias
# TODO SHAP values
# https://pypi.org/project/shap/
# https://mljar.com/blog/feature-importance-in-random-forest/

"""
=========================================
RANDOM FOREST GAP-FILLING FOR TIME SERIES
randomforest_ts
=========================================

This module is part of the diive library:
https://gitlab.ethz.ch/diive/diive

    - Example notebook available in:
        notebooks/GapFilling/RandomForestGapFilling.ipynb

Kudos, optimization of hyper-parameters, grid search
- https://scikit-learn.org/stable/modules/grid_search.html
- https://www.kaggle.com/code/carloscliment/random-forest-regressor-and-gridsearch

"""
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor  # Import the model we are using
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV

import diive.core.dfun.frames as fr
from diive.core.ml.common import MlRegressorGapFillingBase
from diive.pkgs.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class OptimizeParamsRFTS:
    """
    Optimize parameters for random forest model

    """

    def __init__(self,
                 df: DataFrame,
                 target_col: str,
                 **rf_params: dict):
        """Optimize Random Forest hyperparameters using GridSearchCV with time series CV.

        Args:
            df: DataFrame with target and predictor time series (must have complete rows)
            target_col: Column name of target variable
            **rf_params: Parameter ranges to test as lists, e.g.:
                {
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

        Methods:
            optimize(): Run GridSearchCV to find best parameters
            report_optimization(top_n=5): Print comprehensive report with recommendations

        See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        """
        self.model_df = df.copy()
        self.target_col = target_col

        self.regr = RandomForestRegressor()

        self.params = rf_params

        # Attributes
        self._best_params = None
        self._scores = None
        self._cv_results = None
        self._best_score = None
        self._cv_n_splits = None

    @property
    def best_params(self) -> dict:
        """Estimator which gave highest score (or smallest loss if specified) on the left out data"""
        if not self._best_params:
            raise Exception(f'Not available: model scores.')
        return self._best_params

    @property
    def scores(self) -> dict:
        """Return model scores for best model"""
        if not self._scores:
            raise Exception(f'Not available: model scores.')
        return self._scores

    @property
    def cv_results(self) -> DataFrame:
        """Cross-validation results"""
        if not isinstance(self._cv_results, DataFrame):
            raise Exception(f'Not available: cv scores.')
        return self._cv_results

    @property
    def best_score(self) -> float:
        """Mean cross-validated score of the best_estimator"""
        if not self._best_score:
            raise Exception(f'Not available: cv scores.')
        return self._best_score

    @property
    def cv_n_splits(self) -> int:
        """The number of cross-validation splits (folds/iterations)"""
        if not self._cv_n_splits:
            raise Exception(f'Not available: cv scores.')
        return self._cv_n_splits

    def optimize(self):

        y, X, X_names, timestamp = \
            fr.convert_to_arrays(df=self.model_df,
                                 target_col=self.target_col,
                                 complete_rows=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        grid = GridSearchCV(estimator=self.regr,
                            param_grid=self.params,
                            scoring='neg_mean_squared_error',
                            cv=TimeSeriesSplit(n_splits=10),
                            n_jobs=-1)
        grid.fit(X_train, y_train)

        self._cv_results = pd.DataFrame.from_dict(grid.cv_results_)

        # Best parameters after tuning
        # Estimator which gave highest score (or smallest loss if specified) on the left out data
        self._best_params = grid.best_params_

        # Mean cross-validated score of the best_estimator
        self._best_score = grid.best_score_

        # Scorer function used on the held out data to choose the best parameters for the model
        self._scorer = grid.scorer_

        # The number of cross-validation splits (folds/iterations)
        self._cv_n_splits = grid.n_splits_

        grid_predictions = grid.predict(X_test)

        # Stats
        self._scores = prediction_scores(predictions=grid_predictions,
                                         targets=y_test)

    def report_optimization(self, top_n: int = 5) -> None:
        """Print comprehensive optimization report with recommendations.

        Args:
            top_n: Number of top parameter combinations to show (default 5)
        """
        if not self._best_params:
            print("ERROR: Run optimize() first")
            return

        print("\n" + "=" * 80)
        print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION REPORT")
        print("=" * 80)

        # Tested parameter ranges
        print("\n✓ PARAMETER RANGES TESTED")
        print("-" * 80)
        for param, values in sorted(self.params.items()):
            if isinstance(values, list):
                if len(values) > 5:
                    print(f"  {param:<25} : {values[0]} to {values[-1]} ({len(values)} values)")
                else:
                    print(f"  {param:<25} : {values}")
            else:
                print(f"  {param:<25} : {values}")

        # Best parameters section
        print("\n✓ BEST PARAMETERS (GridSearchCV winner)")
        print("-" * 80)
        for param, value in sorted(self._best_params.items()):
            print(f"  {param:<25} = {value}")

        # Best performance section
        print("\n✓ BEST MODEL PERFORMANCE (test set)")
        print("-" * 80)
        print(f"  R² Score              = {self._scores['r2']:>10.4f}  (0-1 scale, higher is better)")
        print(f"  MAE                   = {self._scores['mae']:>10.4f}  (mean absolute error)")
        print(f"  RMSE                  = {self._scores['rmse']:>10.4f}  (root mean squared error)")

        # Top N combinations
        print(f"\n✓ TOP {top_n} PARAMETER COMBINATIONS (by CV score)")
        print("-" * 80)
        top_results = self._cv_results.nsmallest(top_n, 'rank_test_score')
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n  Rank {idx}:")
            mean_score = -row['mean_test_score']  # Negate because neg_mean_squared_error
            print(f"    CV Score: {mean_score:.6f} (lower MSE is better)")
            for param in sorted(self.params.keys()):
                param_key = f'param_{param}'
                if param_key in row:
                    print(f"    {param:<22} = {row[param_key]}")

        # Parameter sensitivity analysis
        print("\n✓ PARAMETER SENSITIVITY (which parameters matter most)")
        print("-" * 80)
        for param in sorted(self.params.keys()):
            param_key = f'param_{param}'
            if param_key in self._cv_results.columns:
                unique_values = self._cv_results[param_key].unique()
                if len(unique_values) > 1:
                    print(f"  {param:<25} : {len(unique_values)} values tested")

        # Recommendation section
        print("\n" + "=" * 80)
        print("RECOMMENDATION FOR PRODUCTION")
        print("=" * 80)
        n_est = self._best_params.get('n_estimators', 100)
        max_d = self._best_params.get('max_depth', None)
        min_split = self._best_params.get('min_samples_split', 5)
        min_leaf = self._best_params.get('min_samples_leaf', 2)
        r2 = self._scores['r2']

        print(f"""
Use these parameters for gap-filling:

    rfts = RandomForestTS(
        input_df=df_engineered,
        target_col='<your_target>',
        n_estimators={n_est},
        max_depth={max_d},
        min_samples_split={min_split},
        min_samples_leaf={min_leaf},
        verbose=1,
        random_state=42
    )

Expected performance: R² ≈ {r2:.4f}
"""
        )
        print("=" * 80 + "\n")


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
            See examples/gap_filling/randomforest_ts.py for complete examples.
            See examples/gap_filling/comparison.py for side-by-side comparison
            with MDS gap-filling.
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
