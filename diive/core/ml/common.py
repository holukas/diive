"""
kudos: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
"""
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay, max_error, median_absolute_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot

import diive.core.dfun.frames as fr
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import include_timestamp_as_cols

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class MlRegressorGapFillingBase:

    def __init__(
            self,
            regressor,
            input_df: DataFrame,
            target_col: str or tuple,
            verbose: bool = True,
            perm_n_repeats: int = 10,
            test_size: float = 0.20,
            features_lag: list = None,
            features_lag_exclude_cols: list = None,
            include_timestamp_as_features: bool = False,
            add_continuous_record_number: bool = False,
            sanitize_timestamp: bool = False,
            random_state: int = None,
            **kwargs
    ):
        """
        Gap-fill timeseries with predictions from random forest model

        Args:
            input_df:
                Contains timeseries of 1 target column and 1+ feature columns.

            target_col:
                Column name of variable in *input_df* that will be gap-filled.

            perm_n_repeats:
                Number of repeats for calculating permutation feature importance.

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

            include_timestamp_as_features:
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
        self.regressor = regressor
        input_df = input_df.copy()
        self.target_col = target_col
        self.perm_n_repeats = perm_n_repeats if perm_n_repeats > 0 else 1
        self.test_size = test_size
        self.features_lag = features_lag
        self.features_lag_exclude_cols = features_lag_exclude_cols
        self.verbose = verbose
        self.include_timestamp_as_features = include_timestamp_as_features
        self.add_continuous_record_number = add_continuous_record_number
        self.sanitize_timestamp = sanitize_timestamp
        self.random_state = random_state
        self.kwargs = kwargs

        # Update model kwargs with random state
        if self.random_state:
            self.kwargs['random_state'] = self.random_state
        else:
            self.kwargs['random_state'] = None

        if self.regressor == RandomForestRegressor:
            self.gfsuffix = '_gfRF'
        elif self.regressor == XGBRegressor:
            self.gfsuffix = '_gfXG'
        else:
            self.gfsuffix = '_gf'

        # Create model dataframe and Add additional data columns
        self.model_df = input_df.copy()
        self.model_df = self._create_additional_datacols()
        self._check_n_cols()

        self.original_input_features = self.model_df.drop(columns=self.target_col).columns.tolist()

        # Create training (80%) and testing dataset (20%)
        # Sort index to keep original order
        _temp_df = self.model_df.copy().dropna()
        self.train_df, self.test_df = train_test_split(_temp_df, test_size=self.test_size,
                                                       random_state=self.random_state, shuffle=True)
        self.train_df = self.train_df.sort_index()
        self.test_df = self.test_df.sort_index()

        self.random_col = None

        # Instantiate model with params
        self._model = self.regressor(**self.kwargs)

        # Attributes
        self._gapfilling_df = None  # Will contain gapfilled target and auxiliary variables
        # self._model = None
        self._traintest_details = dict()
        self._feature_importances = dict()
        self._feature_importances_traintest = pd.DataFrame()
        self._feature_importances_reduction = pd.DataFrame()
        self._scores = dict()
        self._scores_traintest = dict()
        self._accepted_features = []
        self._rejected_features = "None."

        self.n_timeseriessplits = None

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled_col]

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1=gap-filled, 2=gap-filled with fallback"""
        return self.gapfilling_df_[self.target_gapfilled_flag_col]

    @property
    def model_(self):
        """Return model, trained on test data"""
        if not self._model:
            raise Exception(f'Not available: model.')
        return self._model

    @property
    def feature_importances_(self) -> DataFrame:
        """Return feature importance for model used in gap-filling"""
        if not isinstance(self._feature_importances, DataFrame):
            raise Exception(f'Not available: feature importances for gap-filling.')
        return self._feature_importances

    @property
    def feature_importances_traintest_(self) -> DataFrame:
        """Return feature importance from model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_traintest, DataFrame):
            raise Exception(f'Not available: feature importances from training & testing.')
        return self._feature_importances_traintest

    @property
    def feature_importances_reduction_(self) -> pd.DataFrame:
        """Return feature importance from feature reduction, model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_reduction, pd.DataFrame):
            raise Exception(f'Not available: feature importances from feature reduction.')
        return self._feature_importances_reduction

    @property
    def scores_(self) -> dict:
        """Return model scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    @property
    def scores_traintest_(self) -> dict:
        """Return model scores for model trained on training data,
        with scores calculated using test data (holdout set)"""
        if not self._scores_traintest:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores_traintest

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Return gapfilled data and auxiliary variables"""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception(f'Gapfilled data not available.')
        return self._gapfilling_df

    @property
    def traintest_details_(self) -> dict:
        """Return details from train/test splits"""
        if not self._traintest_details:
            raise Exception(f'Not available: details about training & testing.')
        return self._traintest_details

    @property
    def accepted_features_(self) -> list:
        """Return list of accepted features from feature reduction"""
        if not self._accepted_features:
            raise Exception(f'Not available: accepted features from feature reduction.')
        return self._accepted_features

    @property
    def rejected_features_(self) -> list:
        """Return list of rejected features from feature reduction"""
        if not self._rejected_features:
            raise Exception(f'Not available: accepted features from feature reduction.')
        return self._rejected_features

    @staticmethod
    def _fi_across_splits(feature_importances_splits) -> DataFrame:
        """Calculate overall feature importance as mean across splits."""
        fi_columns = [c for c in feature_importances_splits.columns if str(c).endswith("_IMPORTANCE")]
        fi_df = feature_importances_splits[fi_columns].copy()
        fi_df = fi_df.mean(axis=1)
        return fi_df

    def _remove_rejected_features(self, factor: float =1) -> list:
        """Remove features below importance threshold from model dataframe.
        The updated model dataframe will then be used for the next (final) model.
        """

        series = self.feature_importances_reduction_['PERM_IMPORTANCE'].copy()

        # Threshold for feature reduction
        threshold = series.loc[self.random_col]
        threshold = threshold * factor if threshold > 0 else threshold / factor
        print(f">>> Setting threshold for feature rejection to {threshold}.")

        # Get accepted features
        accepted_locs = ((series > threshold) & (series > 0))
        accepted_df = series[accepted_locs].copy()
        accepted_features = accepted_df.index.tolist()

        # # Get rejected features
        # rejected_locs = ((self.feature_importances_reduction_ <= threshold) | (self.feature_importances_reduction_ < 0))
        # fidf_rejected = fi_df.loc[rejected_locs].copy()
        # rejected_features = fidf_rejected.index.tolist()

        # Update dataframe, keep accepted columns
        accepted_cols = [self.target_col]
        accepted_cols = accepted_cols + accepted_features

        return accepted_cols

    @staticmethod
    def _fitmodel(model, X_train, y_train, X_test, y_test):
        """Fit model."""
        if isinstance(model, RandomForestRegressor):
            model.fit(X=X_train, y=y_train)
        elif isinstance(model, XGBRegressor):
            model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
        return model

    def trainmodel(self,
                   showplot_scores: bool = True,
                   showplot_importance: bool = True):
        """
        Train random forest model for gap-filling

        No gap-filling is done here, only the model is trained.

        Args:
            showplot_scores: shows plot of predicted vs observed
            showplot_importance: shows plot of permutation importances

        """

        print("\nTraining final model ...")
        idtxt = f"TRAIN & TEST "

        # Set training and testing data
        y_train = np.array(self.train_df[self.target_col])
        X_train = np.array(self.train_df.drop(self.target_col, axis=1))
        X_test = np.array(self.test_df.drop(self.target_col, axis=1))
        y_test = np.array(self.test_df[self.target_col])
        X_names = self.train_df.drop(self.target_col, axis=1).columns.tolist()

        # Info
        print(f">>> Training model {self.regressor} based on data between "
              f"{self.train_df.index[0]} and {self.train_df.index[-1]} ...")

        # Train the model on training data
        print(f">>> Fitting model to training data ...")
        self._model = self._fitmodel(model=self._model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Predict targets in test data
        print(f">>> Using model to predict target {self.target_col} in unseen test data ...")
        pred_y_test = self.model_.predict(X=X_test)

        # Calculate permutation importance on test data and store in dataframe
        print(f">>> Using model to calculate permutation importance based on unseen test data ...")
        self._feature_importances_traintest = self._permutation_importance(
            model=self.model_, X=X_test, y=y_test, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (permutation importance) ...")
            plot_feature_importance(feature_importances=self.feature_importances_traintest_,
                                    n_perm_repeats=self.perm_n_repeats)

        # Scores
        print(f">>> Calculating prediction scores based on predicting unseen test data of {self.target_col} ...")
        self._scores_traintest = prediction_scores_regr(predictions=pred_y_test, targets=y_test)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values ...")
            plot_observed_predicted(predictions=pred_y_test,
                                    targets=y_test,
                                    scores=self.scores_traintest_,
                                    infotxt=f"{idtxt} trained on training set, tested on test set")

            print(f">>> Plotting residuals and prediction error ...")
            plot_prediction_residuals_error_regr(
                model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                infotxt=f"{idtxt} trained on training set, tested on test set")

        # Collect results
        print(
            f">>> Collecting results, details about training and testing can be accessed by calling .report_traintest().")
        self._traintest_details = dict(
            train_df=self.train_df,
            test_df=self.test_df,
            test_size=self.test_size,
            X_names=X_names,
            model=self.model_,
        )

        print(f">>> Done.")

    def report_traintest(self):
        """Results from model training on test data"""

        idtxt = "MODEL TRAINING & TESTING RESULTS"
        results = self.traintest_details_
        fi = self.feature_importances_traintest_
        test_size_perc = self.test_size * 100
        training_size_perc = 100 - test_size_perc

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"## DATA\n"
            f"  > target: {self.target_col}\n"
            f"  > features: {len(results['X_names'])} {results['X_names']}\n"
            f"  > {len(self.model_df)} records (with missing)\n"
            f"  > {len(self.model_df.dropna())} available records for target and all features (no missing values)\n"
            f"  > training on {len(self.train_df)} records ({training_size_perc:.1f}%) "
            f"of {len(self.train_df)} features between {self.train_df.index[0]} and {self.train_df.index[-1]}\n"
            f"  > testing on {len(self.test_df)} unseen records ({test_size_perc:.1f}%) "
            f"of {self.target_col} between {self.test_df.index[0]} and {self.test_df.index[-1]}\n"
            f"\n"
            f"## MODEL\n"
            f"  > the model was trained on training data ({len(self.train_df)} records)\n"
            f"  > the model was tested on test data ({len(self.test_df)} values)\n"
            f"  > estimator:  {self.model_}\n"
            f"  > parameters:  {self.model_.get_params()}\n"
            f"  > number of features used in model:  {len(results['X_names'])}\n"
            f"  > names of features used in model:  {results['X_names']}\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"  > feature importances were calculated based on unseen test data of {self.target_col} "
            f"({len(self.test_df[self.target_col])} records).\n"
            f"  > feature importances are showing permutation importances from {self.perm_n_repeats} repeats"
            f"\n"
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"\n"
            f"## MODEL SCORES\n"
            f"  All scores were calculated based on unseen test data ({len(self.test_df[self.target_col])} records).\n"
            f"  > MAE:  {self.scores_traintest_['mae']} (mean absolute error)\n"
            f"  > MedAE:  {self.scores_traintest_['medae']} (median absolute error)\n"
            f"  > MSE:  {self.scores_traintest_['mse']} (mean squared error)\n"
            f"  > RMSE:  {self.scores_traintest_['rmse']} (root mean squared error)\n"
            f"  > MAXE:  {self.scores_traintest_['maxe']} (max error)\n"
            f"  > MAPE:  {self.scores_traintest_['mape']:.3f} (mean absolute percentage error)\n"
            f"  > R2:  {self.scores_traintest_['r2']}\n"
        )

    def fillgaps(self,
                 showplot_scores: bool = True,
                 showplot_importance: bool = True):
        """
        Gap-fill data with previously built model

        No new model is built here, instead the last model built in
        the preceding step .trainmodel() is used.

        y = target
        X = features

        """
        self._fillgaps_fullmodel(showplot_scores, showplot_importance)
        self._fillgaps_fallback()
        self._fillgaps_combinepredictions()

    def reduce_features(self, factor:float=1):
        """Reduce number of features using permutation importance

        A random variable is added to features and the permutation importances
        are calculated. The permutation importance of the random variable is the
        benchmark to determine whether a feature is relevant. All features where
        permutation importance is smaller or equal to the importance of the random
        variable are rejected.
        """

        # Info
        print(f"\nFeature reduction based on permutation importance ...")

        df = self.train_df.copy()
        df = df.dropna()

        # Add random variable as feature
        df, self.random_col = self._add_random_variable(df=df)

        X = np.array(df.drop(self.target_col, axis=1))
        y = np.array(df[self.target_col])

        # Instantiate model with params
        model = self.regressor(**self.kwargs)

        # Fit model to training data
        model = self._fitmodel(model=model, X_train=X, y_train=y, X_test=X, y_test=y)

        # https://mljar.com/blog/visualize-tree-from-random-forest/
        # todo from dtreeviz.trees import dtreeviz  # will be used for tree visualization
        # _ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)


        # Calculate permutation importance for all data
        print(f">>> Calculating feature importances (permutation importance, {self.perm_n_repeats} repeats) ...")
        X_names = df.drop(self.target_col, axis=1).columns.tolist()
        feature_importances = self._permutation_importance(model=model, X=X, y=y, X_names=X_names)
        self._feature_importances_reduction = feature_importances.sort_values(by='PERM_IMPORTANCE', ascending=False)

        # Remove variables where mean feature importance across all splits is smaller
        # than or equal to random variable
        # Update dataframe for model building
        accepted_cols = self._remove_rejected_features(factor=factor)

        # Update model data, keep accepted features
        print(">>> Removing rejected features from model data ...")
        self.train_df = self.train_df[accepted_cols].copy()
        self.test_df = self.test_df[accepted_cols].copy()
        self.model_df = self.model_df[accepted_cols].copy()

        self._accepted_features = self.train_df.drop(columns=self.target_col).columns.tolist()
        self._rejected_features = [x for x in self.original_input_features if x not in self.accepted_features_]
        self._rejected_features = 'None.' if not self._rejected_features else self._rejected_features

        # todo OLD from sklearn.model_selection import TimeSeriesSplit
        #  def reduce_features(self, n_timeseriessplits: int = 5):
        #     """Reduce number of features using permutation importance
        #
        #     A random variable is added to features and the permutation importances
        #     are calculated. The permutation importance of the random variable is the
        #     benchmark to determine whether a feature is relevant. All features where
        #     permutation importance is smaller or equal to the importance of the random
        #     variable are rejected.
        #     """
        #
        #     self.n_timeseriessplits = n_timeseriessplits
        #
        #     df = self.train_df.copy()
        #     # df = df.dropna()
        #
        #     # Info
        #     print(f"\nFeature reduction based on permutation importance ...")
        #
        #     # Initialize TimeSeriesSplit for time series cross-validation
        #     tscv = TimeSeriesSplit(n_splits=n_timeseriessplits)
        #
        #     split = 0
        #     feature_importances_splits = DataFrame()  # Collects feature importances per split
        #
        #     # Perform cross-validation using TimeSeriesSplit
        #     for train_index, test_index in tscv.split(df):
        #
        #         split += 1
        #
        #         train_from = df.index[train_index[0]]
        #         train_to = df.index[train_index[-1]]
        #         test_from = df.index[test_index[0]]
        #         test_to = df.index[test_index[-1]]
        #
        #         print(f">>> Working on split {split}: using training data between {train_from} and {train_to}  "
        #               f"to predict target between {test_from} and {test_to}...")
        #
        #         # Add random variable as feature
        #         df, self.random_col = self._add_random_variable(df=df)
        #
        #         train, test = df.iloc[train_index], df.iloc[test_index]
        #
        #         X_train, y_train = train.drop(columns=self.target_col), train[self.target_col]
        #         X_test, y_test = test.drop(columns=self.target_col), test[self.target_col]
        #
        #         # Instantiate model with params
        #         model = self.regressor(**self.kwargs)
        #
        #         # Fit model to training data
        #         model = self._fitmodel(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        #
        #         # Calculate permutation importance for unseen data (test datat) and store in dataframe
        #         X_names = X_test.columns.values
        #         feature_importances_split = self._permutation_importance(
        #             model=model, X=X_test, y=y_test, X_names=X_names)
        #
        #         # Add split info to colnames
        #         names = [f"SPLIT_{split}_{col}" for col in feature_importances_split.columns]
        #         feature_importances_split.columns = names
        #
        #         if split == 1:
        #             feature_importances_splits = feature_importances_split.copy()
        #         else:
        #             feature_importances_splits = pd.concat(
        #                 [feature_importances_splits, feature_importances_split], axis=1)
        #
        #     # Calculate feature importance as the mean across all splits
        #     print(f">>> Calculating overall feature importances as mean across splits ...")
        #     self._feature_importances_reduction = self._fi_across_splits(
        #         feature_importances_splits=feature_importances_splits)
        #     self._feature_importances_reduction = self.feature_importances_reduction_.sort_values(ascending=False)
        #
        #     # Remove variables where mean feature importance across all splits is smaller
        #     # than or equal to random variable
        #     # Update dataframe for model building
        #     accepted_cols = self._remove_rejected_features()
        #
        #
        #     # Update model data, keep accepted features
        #     print(">>> Removing rejected features from model data ...")
        #     self.train_df = self.train_df[accepted_cols].copy()
        #     self.test_df = self.test_df[accepted_cols].copy()
        #     self.model_df = self.model_df[accepted_cols].copy()
        #
        #     self._accepted_features = self.train_df.drop(columns=self.target_col).columns.tolist()
        #     self._rejected_features = [x for x in self.original_input_features if x not in self.accepted_features_]
        #     self._rejected_features = 0 if not self._rejected_features else self._rejected_features
        #
        #     # # todo This could be a way to combine permutation importance with RFECV,
        #     # # but at the time of this writing an import failed (Oct 2023)
        #     # # Train model with random variable included, to detect unimportant features
        #     # df = df.dropna()
        #     # targets = df[self.target_col].copy()
        #     # df = df.drop(self.target_col, axis=1, inplace=False)
        #     # features = df.copy()
        #     # estimator = self.regressor(**self.kwargs)
        #     # splitter = TimeSeriesSplit(n_splits=10)
        #     # from eli5.sklearn import PermutationImportance
        #     # rfecv = RFECV(estimator=PermutationImportance(estimator, scoring='r2', n_iter=10, random_state=self.random_state, cv=splitter),
        #     #               step=1,
        #     #               min_features_to_select=3,
        #     #               cv=splitter,
        #     #               scoring='r2',
        #     #               verbose=self.verbose,
        #     #               n_jobs=-1)
        #     # rfecv.fit(features, targets)
        #     # # Feature importances
        #     # features.drop(features.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
        #     # rfecv_fi_df = pd.DataFrame()
        #     # rfecv_fi_df['FEATURE'] = list(features.columns)
        #     # rfecv_fi_df['IMPORTANCE'] = rfecv.estimator_.feature_importances_
        #     # rfecv_fi_df = rfecv_fi_df.set_index('FEATURE')
        #     # rfecv_fi_df = rfecv_fi_df.sort_values(by='IMPORTANCE', ascending=False)
        #     # # rfecv.cv_results_
        #     # # rfecv.n_features_
        #     # # rfecv.n_features_in_
        #     # # rfecv.ranking_
        #     # # rfecv.support_

    def report_feature_reduction(self):
        """Results from feature reduction"""

        idtxt = "FEATURE REDUCTION"

        # # Original features without random variable
        # _X_names = [x for x in fi.index if x != self.random_col]

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"- Target variable: {self.target_col}\n"
            f"\n"
            f"- The random variable {self.random_col} was added to the original features, "
            f"used as benchmark for detecting relevant feature importances.\n"
            f"\n"
            f"PERMUTATION IMPORTANCE (mean) across all splits of TimeSeriesSplit:\n"
            f"\n"
            f"{self.feature_importances_reduction_}"
            f"\n"
            f"\n"
            f"- These results are from feature reduction. Note that feature importances for "
            f"the final model are calculated during gap-filling.\n"
            f"\n"
            f"--> {len(self.original_input_features)} original input features (before feature reduction): "
            f"{self.original_input_features}\n"
            f"--> {len(self.rejected_features_)} rejected features (during feature reduction): "
            f"{self.rejected_features_}\n"
            f"--> {len(self.accepted_features_)} accepted features (after feature reduction): "
            f"{self.accepted_features_}\n"
        )

    def report_gapfilling(self):
        """Results from gap-filling"""
        # Setup
        idtxt = "GAP-FILLING RESULTS"

        df = self.gapfilling_df_
        model = self.model_
        scores = self.scores_
        fi = self.feature_importances_

        feature_names = fi.index.to_list()
        n_features = len(feature_names)

        locs_observed = df[self.target_gapfilled_flag_col] == 0
        locs_hq = df[self.target_gapfilled_flag_col] == 1
        locs_observed_missing_fromflag = df[self.target_gapfilled_flag_col] > 0
        locs_fallback = df[self.target_gapfilled_flag_col] == 2

        n_observed = locs_observed.sum()
        n_hq = locs_hq.sum()
        n_observed_missing_fromflag = locs_observed_missing_fromflag.sum()
        n_available = len(df[self.target_gapfilled_col].dropna())
        n_potential = len(df.index)
        n_fallback = locs_fallback.sum()
        test_size_perc = self.test_size * 100

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"Model scores and feature importances were calculated from high-quality "
            f"predicted targets ({n_hq} values, {self.target_gapfilled_col} where flag=1) "
            f"in comparison to observed targets ({n_observed} values, {self.target_col}).\n"
            f"\n"
            f"## TARGET\n"
            f"- first timestamp:  {df.index[0]}\n"
            f"- last timestamp:  {df.index[-1]}\n"
            f"- potential number of values: {n_potential} values)\n"
            f"- target column (observed):  {self.target_col}\n"
            f"- missing records (observed):  {df[self.target_col].isnull().sum()} "
            f"(cross-check from flag: {n_observed_missing_fromflag})\n"
            f"- target column (gap-filled):  {self.target_gapfilled_col}  ({n_available} values)\n"
            f"- missing records (gap-filled):  {df[self.target_gapfilled_col].isnull().sum()}\n"
            f"- gap-filling flag: {self.target_gapfilled_flag_col}\n"
            f"  > flag 0 ... observed targets ({n_observed} values)\n"
            f"  > flag 1 ... targets gap-filled with high-quality, all features available ({n_hq} values)\n"
            f"  > flag 2 ... targets gap-filled with fallback ({n_fallback} values)\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"- names of features used in model:  {feature_names}\n"
            f"- number of features used in model:  {n_features}\n"
            f"- permutation importances were calculated from {self.perm_n_repeats} repeats.\n"
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"## MODEL\n"
            f"The model was trained on a training set with test size {test_size_perc:.2f}%.\n"
            f"- estimator:  {model}\n"
            f"- parameters:  {model.get_params()}\n"
            f"\n"
            f"## MODEL SCORES\n"
            f"- MAE:  {scores['mae']} (mean absolute error)\n"
            f"- MedAE:  {scores['medae']} (median absolute error)\n"
            f"- MSE:  {scores['mse']} (mean squared error)\n"
            f"- RMSE:  {scores['rmse']} (root mean squared error)\n"
            f"- MAXE:  {scores['maxe']} (max error)\n"
            f"- MAPE:  {scores['mape']:.3f} (mean absolute percentage error)\n"
            f"- R2:  {scores['r2']}\n"
        )

    def _create_additional_datacols(self) -> pd.DataFrame:
        model_df = self.model_df.copy()

        # Additional data columns
        if any([self.features_lag, self.include_timestamp_as_features,
                self.add_continuous_record_number]):
            print("\nAdding new data columns ...")
            if self.features_lag and (len(model_df.columns) > 1):
                model_df = self._lag_features(features_lag_exclude_cols=self.features_lag_exclude_cols)

            if self.include_timestamp_as_features:
                model_df = include_timestamp_as_cols(df=model_df, txt="")

            if self.add_continuous_record_number:
                model_df = fr.add_continuous_record_number(df=model_df)

        # Timestamp sanitizer
        if self.sanitize_timestamp:
            verbose = True if self.verbose > 0 else False
            tss = TimestampSanitizer(data=model_df, output_middle_timestamp=True, verbose=verbose)
            model_df = tss.get()

        return model_df

    def _permutation_importance(self, model, X, y, X_names) -> DataFrame:
        """Calculate permutation importance"""

        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-feature-importance
        fi = permutation_importance(estimator=model,
                                    X=X, y=y,
                                    n_repeats=self.perm_n_repeats,
                                    random_state=self.random_state,
                                    scoring='r2',
                                    n_jobs=-1)

        # Store permutation importance
        fidf = pd.DataFrame({'PERM_IMPORTANCE': fi.importances_mean,
                             'PERM_SD': fi.importances_std},
                            index=X_names)

        fidf = fidf.sort_values(by='PERM_IMPORTANCE', ascending=False)

        return fidf

    def _add_random_variable(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add random variable as benchmark for relevant feature importances
        random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
        df[random_col] = np.random.RandomState(self.kwargs['random_state']).randn(df.shape[0], 1)
        # df[random_col] = np.random.rand(df.shape[0], 1)
        return df, random_col

    def _lag_features(self, features_lag_exclude_cols):
        """Add lagged variants of variables as new features"""
        exclude_cols = [self.target_col]
        if features_lag_exclude_cols:
            exclude_cols += features_lag_exclude_cols
        return fr.lagged_variants(df=self.model_df,
                                  stepsize=1,
                                  lag=self.features_lag,
                                  exclude_cols=exclude_cols,
                                  verbose=self.verbose)

    def _check_n_cols(self):
        """Check number of columns"""
        if len(self.model_df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.model_df.columns}")

    def _fillgaps_fullmodel(self, showplot_scores, showplot_importance):
        """Apply model to fill missing targets for records where all features are available
        (high-quality gap-filling)"""

        print("\nGap-filling using final model ...")

        # Original input data, contains target and features
        # This dataframe has the full timestamp
        df = self.model_df.copy()

        # Test how the model performs with all y data
        # Since the model was previously trained on test data,
        # here it is checked how well the model performs when
        # predicting all available y data.
        # This is needed to calculate feature importance and scores.
        y, X, X_names, timestamp = fr.convert_to_arrays(
            df=df, target_col=self.target_col, complete_rows=True)

        # Predict all targets (no test split)
        print(f">>> Using final model on all data to predict target {self.target_col} ...")
        pred_y = self.model_.predict(X=X)

        # Calculate permutation importance and store in dataframe
        print(f">>> Using final model on all data to calculate permutation importance ...")
        self._feature_importances = self._permutation_importance(
            model=self._model, X=X, y=y, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (permutation importance) ...")
            plot_feature_importance(feature_importances=self.feature_importances_,
                                    n_perm_repeats=self.perm_n_repeats)

        # Scores, using all targets
        print(f">>> Calculating prediction scores based on all data predicting {self.target_col} ...")
        self._scores = prediction_scores_regr(predictions=pred_y, targets=y)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values based on all data ...")
            plot_observed_predicted(predictions=pred_y,
                                    targets=y,
                                    scores=self.scores_,
                                    infotxt=f"trained on training set, tested on FULL set")

            # print(f">>> Plotting residuals and prediction error based on all data ...")
            # plot_prediction_residuals_error_regr(
            #     model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            #     infotxt=f"trained on training set, tested on full set")

        # In the next step, all available features are used to
        # predict the target for records where all features are available.
        # Feature data for records where all features are available:
        print(f">>> Predicting target {self.target_col} where all features are available ...", end=" ")
        features_df = df.drop(self.target_col, axis=1)  # Remove target data
        features_df = features_df.dropna()  # Keep rows where all features available
        X = features_df.to_numpy()  # Features are needed as numpy array
        feature_names = features_df.columns.tolist()

        # Predict targets for all records where all features are available
        pred_y = self.model_.predict(X=X)
        print(f"predicted {len(pred_y)} records.")

        # Collect gapfilling results in df
        # Define column names for gapfilled_df
        print(">>> Collecting results for final model ...")
        self._define_cols()

        # Collect predictions in dataframe
        self._gapfilling_df = pd.DataFrame(data={self.pred_fullmodel_col: pred_y}, index=features_df.index)

        # Add target to dataframe
        self._gapfilling_df[self.target_col] = df[self.target_col].copy()

        # Gap locations
        # Make column that contains predicted values
        # for rows where target is missing
        _gap_locs = self._gapfilling_df[self.target_col].isnull()  # Locations where target is missing
        self._gapfilling_df[self.pred_gaps_col] = self._gapfilling_df.loc[
            _gap_locs, self.pred_fullmodel_col]

        # Flag
        # Make flag column that indicates where predictions for
        # missing targets are available, where 0=observed, 1=gapfilled
        # todo Note that missing predicted gaps = 0. change?
        _gapfilled_locs = self._gapfilling_df[self.pred_gaps_col].isnull()  # Non-gapfilled locations
        _gapfilled_locs = ~_gapfilled_locs  # Inverse for gapfilled locations
        self._gapfilling_df[self.target_gapfilled_flag_col] = _gapfilled_locs
        self._gapfilling_df[self.target_gapfilled_flag_col] = self._gapfilling_df[
            self.target_gapfilled_flag_col].astype(
            int)

        # Gap-filled time series
        # Fill missing records in target with predicions
        n_missing = self._gapfilling_df[self.target_col].isnull().sum()
        print(f">>> Filling {n_missing} missing records in target with predictions from final model ...")
        print(f">>> Storing gap-filled time series in variable {self.target_gapfilled_col} ...")
        self._gapfilling_df[self.target_gapfilled_col] = \
            self._gapfilling_df[self.target_col].fillna(self._gapfilling_df[self.pred_fullmodel_col])

        # Restore original full timestamp
        print(">>> Restoring original timestamp in results ...")
        self._gapfilling_df = self._gapfilling_df.reindex(df.index)

        # SHAP values
        # https://pypi.org/project/shap/
        # https://mljar.com/blog/feature-importance-in-random-forest/

    def _fillgaps_fallback(self):

        # Fallback gapfilling
        # Fill still existing gaps in full timestamp data
        # Build fallback model exclusively from timestamp features.
        # Here, the model is trained on the already gapfilled time series,
        # using info from the timestamp, e.g. DOY
        _still_missing_locs = self._gapfilling_df[self.target_gapfilled_col].isnull()
        _num_still_missing = _still_missing_locs.sum()  # Count number of still-missing values
        if _num_still_missing > 0:

            print(f"\nGap-filling {_num_still_missing} remaining missing records in "
                  f"{self.target_gapfilled_col} using fallback model ...")
            print(f">>> Fallback model is trained on {self.target_gapfilled_col} and timestamp info ...")

            fallback_predictions, \
                fallback_timestamp = \
                self._predict_fallback(series=self._gapfilling_df[self.target_gapfilled_col])

            fallback_series = pd.Series(data=fallback_predictions, index=fallback_timestamp)
            self._gapfilling_df[self.pred_fallback_col] = fallback_series
            self._gapfilling_df[self.target_gapfilled_col] = \
                self._gapfilling_df[self.target_gapfilled_col].fillna(fallback_series)

            self._gapfilling_df.loc[_still_missing_locs, self.target_gapfilled_flag_col] = 2  # Adjust flag, 2=fallback
        else:
            self._gapfilling_df[self.pred_fallback_col] = None

        # Cumulative
        self._gapfilling_df[self.target_gapfilled_cumu_col] = \
            self._gapfilling_df[self.target_gapfilled_col].cumsum()

    def _fillgaps_combinepredictions(self):
        """Combine predictions of full model with fallback predictions"""
        print(">>> Combining predictions from full model and fallback model ...")
        # First add predictions from full model
        self._gapfilling_df[self.pred_col] = self._gapfilling_df[self.pred_fullmodel_col].copy()
        # Then fill remaining gaps with predictions from fallback model
        self._gapfilling_df[self.pred_col] = (
            self._gapfilling_df[self.pred_col].fillna(self._gapfilling_df[self.pred_fallback_col]))

    def _predict_fallback(self, series: pd.Series):
        """Fill data gaps using timestamp features only, fallback for still existing gaps"""
        gf_fallback_df = pd.DataFrame(series)
        gf_fallback_df = include_timestamp_as_cols(df=gf_fallback_df, txt="(ONLY FALLBACK)")

        # Build model for target predictions *from timestamp*
        y_fallback, X_fallback, _, _ = \
            fr.convert_to_arrays(df=gf_fallback_df,
                                 target_col=self.target_gapfilled_col,
                                 complete_rows=True)

        # Instantiate new model with same params as before
        model_fallback = self.regressor(**self.kwargs)

        # Train the model on all available records ...
        model_fallback = self._fitmodel(model=model_fallback, X_train=X_fallback, y_train=y_fallback, X_test=X_fallback,
                                        y_test=y_fallback)
        # model_fallback.fit(X=X_fallback, y=y_fallback)

        # ... and use it to predict all records for full timestamp
        full_timestamp_df = gf_fallback_df.drop(self.target_gapfilled_col, axis=1)  # Remove target data
        X_fallback_full = full_timestamp_df.to_numpy()  # Features are needed as numpy array

        print(f">>> Predicting target {self.target_gapfilled_col} using fallback model ...")
        pred_y_fallback = model_fallback.predict(X=X_fallback_full)  # Predict targets in test data
        full_timestamp = full_timestamp_df.index

        return pred_y_fallback, full_timestamp

    def _results(self, gapfilled_df, most_important_df, model_r2, still_missing_locs):
        """Summarize gap-filling results"""

        _vals_max = len(gapfilled_df.index)
        _vals_before = len(gapfilled_df[self.target_col].dropna())
        _vals_after = len(gapfilled_df[self.target_gapfilled_col].dropna())
        _vals_fallback_filled = still_missing_locs.sum()
        _perc_fallback_filled = (_vals_fallback_filled / _vals_max) * 100

        print(f"Gap-filling results for {self.target_col}\n"
              f"max possible: {_vals_max} values\n"
              f"before gap-filling: {_vals_before} values\n"
              f"after gap-filling: {_vals_after} values\n"
              f"gap-filled with fallback: {_vals_fallback_filled} values / {_perc_fallback_filled:.1f}%\n"
              f"used features:\n{most_important_df}\n"
              f"predictions vs targets, R2 = {model_r2:.3f}")

    def _define_cols(self):
        self.pred_col = ".PREDICTIONS"
        self.pred_fullmodel_col = ".PREDICTIONS_FULLMODEL"
        self.pred_fallback_col = ".PREDICTIONS_FALLBACK"
        self.pred_gaps_col = ".GAP_PREDICTIONS"
        self.target_gapfilled_col = f"{self.target_col}{self.gfsuffix}"
        self.target_gapfilled_flag_col = f"FLAG_{self.target_gapfilled_col}_ISFILLED"  # "[0=measured]"
        self.target_gapfilled_cumu_col = ".GAPFILLED_CUMULATIVE"


# import pandas as pd
# from numpy import ndarray
# from sklearn.ensemble import RandomForestRegressor  # Import the model we are using
# from sklearn.inspection import permutation_importance
# def feature_importances(estimator: self.regressor,
#                         X: ndarray,
#                         y: ndarray,
#                         model_feature_names: list,
#                         perm_n_repeats: int = 10,
#                         random_col: str = None,
#                         showplot: bool = True,
#                         verbose: int = 1) -> dict:
#     """
#     Calculate feature importance, based on built-in method and permutation
#
#     The built-in method for RandomForestRegressor() is Gini importance.
#
#
#     See:
#     - https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
#
#     Args:
#         estimator: fitted estimator
#         X: features to predict y, required for permutation importance
#         y: targets, required for permutation importance
#         model_feature_names: list
#         perm_n_repeats: number of repeats for computing permutation importance
#         random_col: name of the random variable used as benchmark for relevant importance results
#         showplot: shows plot of permutation importance results
#         verbose: print details
#
#     Returns:
#         list of recommended features where permutation importance was higher than random, and
#         two dataframes with overview of filtered and unfiltered importance results, respectively
#     """
#     # Store built-in feature importance (Gini)
#     importances_gini_df = pd.DataFrame({'GINI_IMPORTANCE': estimator.feature_importances_},
#                                        index=model_feature_names)
#
#     # Calculate permutation importance
#     # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-feature-importance
#     perm_results = permutation_importance(estimator, X, y, n_repeats=perm_n_repeats, random_state=self.random_state,
#                                           scoring='r2', n_jobs=-1)
#
#     # Store permutation importance
#     importances_perm_df = pd.DataFrame({'PERM_IMPORTANCE': perm_results.importances_mean,
#                                         'PERM_SD': perm_results.importances_std},
#                                        index=model_feature_names)
#
#     # Store importance in one df
#     importances_df = pd.concat([importances_perm_df, importances_gini_df], axis=1)
#     importances_df = importances_df.sort_values(by='PERM_IMPORTANCE', ascending=True)
#
#     # Keep features with higher permutation importance than random variable
#     rejected_features = []
#     if random_col:
#         perm_importance_threshold = importances_df['PERM_IMPORTANCE'][random_col]
#         filtered_importances_df = \
#             importances_df.loc[importances_df['PERM_IMPORTANCE'] > perm_importance_threshold].copy()
#
#         # Get list of recommended features where permutation importance is larger than random
#         recommended_features = filtered_importances_df.index.tolist()
#
#         # Find rejected features below the importance threshold
#         before_cols = importances_df.index.tolist()
#         after_cols = filtered_importances_df.index.tolist()
#
#         for item in before_cols:
#             if item not in after_cols:
#                 rejected_features.append(item)
#
#         if verbose > 0:
#             print(f"Accepted variables: {after_cols}  -->  "
#                   f"above permutation importance threshold of {perm_importance_threshold}")
#             print(f"Rejected variables: {rejected_features}  -->  "
#                   f"below permutation importance threshold of {perm_importance_threshold}")
#     else:
#         # No random variable considered
#         perm_importance_threshold = None
#         recommended_features = importances_df.index.tolist()
#         filtered_importances_df = importances_df.copy()
#
#     if showplot:
#         fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
#
#         importances_df['PERM_IMPORTANCE'].plot.barh(color='#008bfb', yerr=importances_df['PERM_SD'], ax=axs[0])
#         axs[0].set_xlabel("Permutation importance")
#         axs[0].set_ylabel("Feature")
#         axs[0].set_title("Permutation importance")
#         axs[0].legend(loc='lower right')
#
#         importances_df['GINI_IMPORTANCE'].plot.barh(color='#008bfb', ax=axs[1])
#         axs[1].set_xlabel("Gini importance")
#         axs[1].set_title("Built-in importance (Gini)")
#         axs[1].legend(loc='lower right')
#
#         if random_col:
#             # Check Gini importance of random variable, used for display purposes only (plot)
#             gini_importance_threshold = importances_df['GINI_IMPORTANCE'][random_col]
#             axs[0].axvline(perm_importance_threshold, color='#ff0051', ls='--', label="importance threshold")
#             axs[1].axvline(gini_importance_threshold, color='#ff0051', ls='--', label="importance threshold")
#
#         fig.tight_layout()
#         fig.show()
#
#     importances = {
#         'recommended_features': recommended_features,
#         'rejected_features': rejected_features,
#         'filtered_importances': filtered_importances_df,
#         'importances': importances_df
#     }
#
#     return importances


def prediction_scores_regr(predictions: np.array,
                           targets: np.array) -> dict:
    """
    Calculate prediction scores for regression estimator

    See:
    - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    # Calculate stats
    scores = {
        'mae': mean_absolute_error(targets, predictions),
        'medae': median_absolute_error(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': root_mean_squared_error(targets, predictions),
        'mape': mean_absolute_percentage_error(targets, predictions),
        'maxe': max_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }
    return scores


def plot_feature_importance(feature_importances: pd.DataFrame, n_perm_repeats: int):
    fig, axs = plt.subplots(ncols=1, figsize=(9, 16))
    _fidf = feature_importances.copy().sort_values(by='PERM_IMPORTANCE', ascending=True)
    _fidf['PERM_IMPORTANCE'].plot.barh(color='#008bfb', yerr=_fidf['PERM_SD'], ax=axs)
    axs.set_xlabel("Feature importance")
    axs.set_ylabel("Feature")
    axs.set_title(f"Permutation importance ({n_perm_repeats} permutations)")
    axs.legend(loc='lower right')
    fig.tight_layout()
    fig.show()


def plot_observed_predicted(targets: np.ndarray,
                            predictions: np.ndarray,
                            scores: dict,
                            infotxt: str = "",
                            random_state: int = None):
    # Plot observed and predicted
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(targets,
                                            y_pred=predictions,
                                            kind="actual_vs_predicted",
                                            subsample=None,
                                            ax=axs[0],
                                            random_state=random_state)
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(targets,
                                            y_pred=predictions,
                                            kind="residual_vs_predicted",
                                            subsample=None,
                                            ax=axs[1],
                                            random_state=random_state)
    axs[1].set_title("Residuals vs. Predicted Values")
    n_vals = len(predictions)
    fig.suptitle(f"Plotting cross-validated predictions ({infotxt})\n"
                 f"n_vals={n_vals}, MAE={scores['mae']:.3f}, RMSE={scores['rmse']:.3f}, r2={scores['r2']:.3f}")
    plt.tight_layout()
    plt.show()


def plot_prediction_residuals_error_regr(model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         infotxt: str):
    """
    Plot residuals and prediction error

    Args:
        model:
        X_train: predictors in training data
        y_train: targets in training data
        X_test: predictors in test data
        y_test: targets in test data
        infotxt: text displayed in figure header

    Kudos:
    - https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
    - https://www.scikit-yb.org/en/latest/api/regressor/peplot.html

    """

    # fig, axs = plt.subplots(ncols=2, figsize=(14, 4))
    # fig, ax = plt.subplots()

    # Histogram can be replaced with a Q-Q plot, which is a common way
    # to check that residuals are normally distributed. If the residuals
    # are normally distributed, then their quantiles when plotted against
    # quantiles of normal distribution should form a straight line.
    fig, ax = plt.subplots()
    fig.suptitle(f"{infotxt}")
    vis = ResidualsPlot(model, hist=False, qqplot=True, ax=ax)
    vis.fit(X_train, y_train)  # Fit the training data to the visualizer
    vis.score(X_test, y_test)  # Evaluate the model on the test data
    vis.show()  # Finalize and render the figure

    # difference between the observed value of the target variable (y)
    # and the predicted value (ŷ), i.e. the error of the prediction
    fig, ax = plt.subplots()
    fig.suptitle(f"{infotxt}")
    vis = PredictionError(model)
    vis.fit(X_train, y_train)  # Fit the training data to the visualizer
    vis.score(X_test, y_test)  # Evaluate the model on the test data
    vis.show()

    # fig.suptitle(f"{infotxt}")
    # plt.tight_layout()
    # fig.show()
