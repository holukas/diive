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
import numpy as np
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
        """
        Args:
            df: dataframe of target and predictor time series
            target_col: name of target in *df*, all variables that are not *target* are
                used as predictors
            **rf_params: dict of parameters for random forest model, where parameter ranges are
                provided as lists, e.g.
                    rf_params = {
                        'n_estimators': list(range(2, 12, 2)),
                        'criterion': ['root_mean_squared_error'],
                        'max_depth': [None],
                        'min_samples_split': list(range(2, 12, 2)),
                        'min_samples_leaf': [1, 3, 6]
                    }

                For an overview of RF parameters see:
                https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
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


class RandomForestTS(MlRegressorGapFillingBase):

    def __init__(self,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: bool = True,
                 test_size: float = 0.25,
                 features_lag: list = None,
                 features_lag_stepsize: int = 1,
                 features_lag_exclude_cols: list = None,
                 features_rolling: list = None,
                 features_rolling_exclude_cols: list = None,
                 features_rolling_stats: list = None,
                 features_diff: list = None,
                 features_diff_exclude_cols: list = None,
                 features_poly_degree: int = None,
                 features_poly_exclude_cols: list = None,
                 vectorize_timestamps: bool = False,
                 add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False,
                 **kwargs):
        """
        Gap-fill timeseries with predictions from random forest model

        Args:
            input_df:
                Contains timeseries of 1 target column and 1+ feature columns.

            target_col:
                Column name of variable in *input_df* that will be gap-filled.

            test_size:
                Proportion of the dataset to include in the test split,
                between 0.0 and 1.0.

            features_lag:
                List of integers (number of records), includes lagged variants of predictors.
                If features_lag=None, no lagged variants are added.
                Example:
                    - features_lag=[-2, +2] includes variants that are lagged by -2, -1, +1 and
                    +2 records in the dataset, for each feature already present in the data.
                     For a variable named *TA*, this created the following output:
                    TA    = [  5,   6,   7, 8  ]
                    TA-2  = [NaN, NaN,   5, 6  ]
                    TA-1  = [NaN,   5,   6, 7  ]  --> each TA record is paired with the preceding record TA-1
                    TA+1  = [  6,   7,   8, NaN]  --> each TA record is paired with the next record TA+1
                    TA+2  = [  7,   8, NaN, NaN]

            features_lag_exclude_cols:
                List of predictors for which no lagged variants are added.
                Example: with ['A', 'B'] no lagged variants for variables 'A' and 'B' are added.

            features_rolling:
                List of window sizes (in records) for rolling statistics.
                For each window size, rolling mean and rolling std are added for every
                feature column. If None, no rolling statistics are added.
                Example: features_rolling=[6, 48] with 30-min data adds 3-hour and 24-hour
                rolling mean and std for each driver variable.

            features_rolling_exclude_cols:
                List of column names excluded from rolling statistics.
                Example: ['Rg_f'] skips rolling features for Rg_f.

            features_rolling_stats:
                List of additional rolling statistics to compute beyond mean and std.
                Options: 'median', 'min', 'max', 'std', 'q25', 'q75'
                If None, only mean and std are computed.
                Example: features_rolling_stats=['median', 'min', 'max', 'q25', 'q75']

            features_diff:
                List of integer difference orders for temporal momentum features.
                For each order, creates `.{col}_DIFF{order}` columns.
                Example: features_diff=[1, 2] creates 1st and 2nd order differences.
                If None, no differencing is applied.

            features_diff_exclude_cols:
                List of column names excluded from differencing.
                Example: ['RECORD_NUMBER'] skips differencing for continuous record number.

            features_poly_degree:
                Polynomial degree for feature expansion (e.g., 2 for squared terms).
                If None, no polynomial features are added.
                Creates features like `.{col}_POL2` for degree 2, `.{col}_POL3` for degree 3.
                Example: features_poly_degree=2 creates squared terms for all driver variables.

            features_poly_exclude_cols:
                List of column names excluded from polynomial expansion.
                Example: ['RECORD_NUMBER'] skips polynomial features for continuous record number.

            vectorize_timestamps:
                Include timestamp info as integer data: year, season, month, week, doy, hour

            add_continuous_record_number:
                Add continuous record number as new column

            sanitize_timestamp:
                Validate and prepare timestamps for further processing

        Attributes:
            gapfilled_df
            - .PREDICTIONS_FULLMODEL uses the output from the full RF model where
              all features where available.
            - .PREDICTIONS_FALLBACK uses the output from the fallback RF model, which
              was trained on the combined observed + .PREDICTIONS_FULLMODEL data, using
              only the timestamp info as features.
        """

        # Args
        super().__init__(
            regressor=RandomForestRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            test_size=test_size,
            features_lag=features_lag,
            features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff,
            features_diff_exclude_cols=features_diff_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            **kwargs
        )


class QuickFillRFTS:
    """
    Quick gap-filling using RandomForestTS with pre-defined minimal parameters

    The purpose of this class is preliminary/exploratory gap-filling for quick tests
    and rapid prototyping. It is NOT meant for production/final gap-filling.

    Uses minimal feature engineering and model complexity for speed:
    - n_estimators=3 (very fast, low quality)
    - Single lag: [-1, -1] (only immediate past value)
    - No rolling statistics, differencing, or timestamp features
    - Shallow trees with larger min_samples for fast inference

    Attributes:
        gapfilling_df(): DataFrame with gap-filled target and flags
        get_gapfilled_target(): Series with gap-filled values
        get_flag(): Series with gap-filling flags (0=observed, 1=filled, 2=fallback)
    """

    def __init__(self, df: DataFrame, target_col: str or tuple):
        self.df = df.copy()
        self.target_col = target_col
        self.rfts = None

        # Minimal parameters for fast, uncomplicated gap-filling
        self.rfts = RandomForestTS(
            input_df=self.df,
            target_col=self.target_col,
            verbose=True,
            features_lag=[-1, -1],  # Only immediate past value
            features_lag_stepsize=1,
            vectorize_timestamps=False,  # No timestamp features (speed)
            add_continuous_record_number=False,  # No extra features
            sanitize_timestamp=False,  # Skip validation (speed)
            n_estimators=3,  # Minimal trees for speed
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


def _example_quickfill():
    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Setup, imports
    import numpy as np
    import importlib.metadata
    from datetime import datetime
    from diive.configs.exampledata import load_exampledata_parquet
    from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    from diive.core.dfun.stats import sstats  # Time series stats
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # from diive.pkgs.gapfilling.randomforest_ts import QuickFillRFTS
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"This page was last modified on: {dt_string}")
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")

    # Show docstring for QuickFillRFTS
    print(QuickFillRFTS.__name__)
    print(QuickFillRFTS.__doc__)

    # Example data
    df = load_exampledata_parquet()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    lowquality = df["QCF_NEE"] > 0
    df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()
    df.describe()
    statsdf = sstats(df[TARGET_COL])
    print(statsdf)
    TimeSeries(series=df[TARGET_COL]).plot()

    # QuickFill example
    qf = QuickFillRFTS(df=df, target_col=TARGET_COL)
    qf.fill()
    qf.report()
    gapfilled = qf.get_gapfilled_target()

    # Plot
    HeatmapDateTime(series=df[TARGET_COL]).show()
    HeatmapDateTime(series=gapfilled).show()


def _example_rfts():
    """Example: Random Forest gap-filling with minimal feature engineering"""
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
    from diive.configs.exampledata import load_exampledata_parquet
    df_orig = load_exampledata_parquet()
    df = df_orig.copy()
    keep = (df.index.year >= 2020) & (df.index.year <= 2020)
    df = df[keep].copy()
    df = df[subsetcols].copy()

    # Random forest with minimal parameters for speed
    rfts = RandomForestTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=1,
        features_lag=[-1, -1],
        features_lag_stepsize=1,
        features_lag_exclude_cols=None,
        features_rolling=None,
        features_rolling_exclude_cols=None,
        features_rolling_stats=None,
        features_diff=None,
        features_diff_exclude_cols=None,
        features_poly_degree=None,
        features_poly_exclude_cols=None,
        vectorize_timestamps=False,
        add_continuous_record_number=False,
        sanitize_timestamp=False,
        n_estimators=3,
        random_state=42,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )

    # Feature reduction using SHAP importance
    rfts.reduce_features(shap_threshold_factor=0.5)
    rfts.report_feature_reduction()

    # Train model
    rfts.trainmodel(showplot_scores=False, showplot_importance=False)
    rfts.report_traintest()

    # Gap-fill data
    rfts.fillgaps(showplot_scores=False, showplot_importance=False)
    rfts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = rfts.get_gapfilled_target()

    print("Finished.")


def _example_optimize():
    from diive.configs.exampledata import load_exampledata_parquet

    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    df = load_exampledata_parquet()
    subset = df[subsetcols].copy()
    _subset = df.index.year >= 2020
    subset = subset[_subset].copy()

    # Random forest parameters
    rf_params = {
        'n_estimators': list(range(10, 20, 2)),
        # 'n_estimators': list(range(100, 12, 2)),
        'criterion': ['squared_error'],
        'max_depth': [None],
        'min_samples_split': [2, 4, 8, 16],
        # 'min_samples_split': list(range(2, 12, 2)),
        'min_samples_leaf': [1, 2, 4, 8],

        # 'min_samples_leaf': list(range(1, 6, 1))
    }

    # Optimization
    opt = OptimizeParamsRFTS(
        df=subset,
        target_col=TARGET_COL,
        **rf_params
    )

    opt.optimize()

    print(opt.best_params)
    print(opt.scores['r2'])
    # print(opt.cv_results)

    best = opt.cv_results.copy()
    best = best.sort_values(by='rank_test_score')
    print(best)
    # print(best.iloc[0].T)

    # print(opt.best_score)
    # print(opt.cv_n_splits)


if __name__ == '__main__':
    _example_quickfill()
    # _example_rfts()
    # _example_optimize()
