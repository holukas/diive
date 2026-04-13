"""
kudos: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot

import diive.core.dfun.frames as fr
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import vectorize_timestamps
from diive.pkgs.createvar.laggedvariants import lagged_variants
from diive.pkgs.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class MlRegressorGapFillingBase:

    def __init__(self,
                 regressor,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: int = 0,
                 features_lag: list[int, int] = None,
                 features_lag_stepsize: int = 1,
                 features_lag_exclude_cols: list = None,
                 features_rolling: list = None,
                 features_rolling_exclude_cols: list = None,
                 vectorize_timestamps: bool = False,
                 add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False,
                 test_size: float = 0.25,
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
                feature column (excluding target and any cols in features_rolling_exclude_cols).
                If None, no rolling statistics are added.
                Example: features_rolling=[6, 48] with 30-min data adds 3-hour and 24-hour
                rolling mean and std for each driver variable.
                Column naming: '.{col}_mean{w}' and '.{col}_std{w}', e.g. '.Tair_f_mean6'.

            features_rolling_exclude_cols:
                List of column names excluded from rolling statistics.
                Example: ['Rg_f'] skips rolling features for Rg_f.

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
        self.regressor = regressor
        input_df = input_df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.features_lag = features_lag
        self.features_lag_stepsize = features_lag_stepsize
        self.features_lag_exclude_cols = features_lag_exclude_cols
        self.features_rolling = features_rolling
        self.features_rolling_exclude_cols = features_rolling_exclude_cols
        self.verbose = verbose
        self.vectorize_timestamps = vectorize_timestamps
        self.add_continuous_record_number = add_continuous_record_number
        self.sanitize_timestamp = sanitize_timestamp
        self.kwargs = kwargs

        self._random_state = self.kwargs['random_state'] if 'random_state' in self.kwargs else None

        if self.regressor == RandomForestRegressor:
            self.gfsuffix = '_gfRF'
        elif self.regressor == XGBRegressor:
            self.gfsuffix = '_gfXG'
        else:
            self.gfsuffix = '_gf'

        if verbose:
            print(f"\n\n{'=' * 60}\nStarting gap-filling for\n{self.target_col}\nusing {self.regressor}\n{'=' * 60}")

        # Create model dataframe and Add additional data columns
        self.model_df = input_df.copy()

        # Original input features (all features except target)
        self.original_input_features = self.model_df.drop(columns=self.target_col).columns.tolist()

        # Create additional data columns
        self.model_df = self._create_additional_datacols()

        self._check_n_cols()

        # Check if features complete
        n_vals_index = len(self.model_df.index)
        fstats = self.model_df[self.original_input_features].describe()
        not_complete = fstats.loc['count'] < n_vals_index
        not_complete = not_complete[not_complete].index.tolist()
        if len(not_complete) > 0:
            print(f"(!)Some features are incomplete and have less than {n_vals_index} values:")
            for nc in not_complete:
                print(f"    --> {nc} ({fstats[nc]['count'].astype(int)} values)")
            print(f"This means that not all target values can be predicted based on the full model.")

        # Create training (75%) and testing dataset (25%)
        # Sort index to keep original order
        _temp_df = self.model_df.copy().dropna()

        self.train_df, self.test_df = train_test_split(_temp_df, test_size=self.test_size,
                                                       random_state=self._random_state, shuffle=True)

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
        self._rejected_features = []

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

    def _remove_rejected_features(self, factor: float = 1, infotxt="[ FEATURE REDUCTION ]") -> list:
        """Remove features that are below importance threshold or that are below
        zero from model dataframe. The updated model dataframe will then be used
        for the next (final) model.
        """

        series = self.feature_importances_reduction_['SHAP_IMPORTANCE'].copy()

        # Threshold for feature reduction
        threshold = series.loc[self.random_col]
        threshold = threshold * factor if threshold > 0 else threshold / factor
        print(f"{infotxt} >>> Setting threshold for feature rejection to {threshold}.")

        # Get accepted features
        accepted_locs = ((series > threshold) & (series > 0))
        accepted_df = pd.DataFrame(series[accepted_locs])
        accepted_features = accepted_df.index.tolist()
        print(f"\n{infotxt} >>> Accepted features and their importance:\n{accepted_df}")

        # Get rejected features (everything not accepted, incl. boundary and random col)
        rejected_locs = ~accepted_locs
        rejected_df = pd.DataFrame(series[rejected_locs])
        rejected_features = rejected_df.index.tolist()
        print(f"\n{infotxt} >>> Rejected features and their importance:\n{rejected_df}")

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
            showplot_importance: shows plot of SHAP feature importances

        """

        print("\nTraining final model ...")
        idtxt = f"TRAIN & TEST "

        # Set training and testing data
        train_df = self.train_df.copy()
        y_train = np.array(train_df[self.target_col])
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

        # Calculate SHAP-based feature importance on test data and store in dataframe
        print(f">>> Using model to calculate SHAP feature importance based on unseen test data ...")
        self._feature_importances_traintest = self._shap_importance(
            model=self.model_, X=X_test, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (SHAP) ...")
            plot_feature_importance(feature_importances=self.feature_importances_traintest_)

        # Scores
        print(f">>> Calculating prediction scores based on predicting unseen test data of {self.target_col} ...")
        self._scores_traintest = prediction_scores(predictions=pred_y_test, targets=y_test)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values ...")
            plot_observed_predicted(predictions=pred_y_test,
                                    targets=y_test,
                                    scores=self.scores_traintest_,
                                    infotxt=f"{idtxt} trained on training set, tested on test set",
                                    random_state=self._random_state)

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
            f"  > feature importances show mean absolute SHAP values.\n"
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

    def reduce_features(self, factor: float = 1):
        """Reduce number of features using SHAP importance

        A random variable is added to features and SHAP importances
        are calculated. The SHAP importance of the random variable is the
        benchmark to determine whether a feature is relevant. All features where
        SHAP importance is smaller or equal to the importance of the random
        variable are rejected.
        """

        infotxt = "[ FEATURE REDUCTION ]"

        # Info
        print(f"\n{infotxt} Feature reduction based on SHAP importance ...")

        df = self.train_df.copy()
        df = df.dropna()

        # Add random variable as feature
        df, self.random_col = self._add_random_variable(df=df)

        X = np.array(df.drop(self.target_col, axis=1))
        y = np.array(df[self.target_col])

        # Instantiate model with params
        model = self.regressor(**self.kwargs)

        model.get_params()

        # Fit model to training data
        model = self._fitmodel(model=model, X_train=X, y_train=y, X_test=X, y_test=y)

        # https://mljar.com/blog/visualize-tree-from-random-forest/
        # todo from dtreeviz.trees import dtreeviz  # will be used for tree visualization
        # _ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)

        # Calculate SHAP importance for all data
        print(f"{infotxt} >>> Calculating feature importances (SHAP) ...")
        X_names = df.drop(self.target_col, axis=1).columns.tolist()
        feature_importances = self._shap_importance(model=model, X=X, X_names=X_names)
        self._feature_importances_reduction = feature_importances.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        # Remove variables where mean feature importance across all splits is smaller
        # than or equal to random variable
        # Update dataframe for model building
        accepted_cols = self._remove_rejected_features(factor=factor)

        # Update model data, keep accepted features
        print(f"{infotxt} >>> Removing rejected features from model data ...")
        self.train_df = self.train_df[accepted_cols].copy()
        self.test_df = self.test_df[accepted_cols].copy()
        self.model_df = self.model_df[accepted_cols].copy()

        self._accepted_features = self.train_df.drop(columns=self.target_col).columns.tolist()
        self._rejected_features = [x for x in self.original_input_features if x not in self.accepted_features_]
        self._rejected_features = 'None.' if not self._rejected_features else self._rejected_features

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
            f"SHAP IMPORTANCE (mean absolute SHAP values):\n"
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
            f"- feature importances calculated using SHAP (TreeExplainer).\n"
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

    def _create_lagged_variants(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add lagged features because there are no original features.")
        _out_df = lagged_variants(df=work_df,
                                  stepsize=self.features_lag_stepsize,
                                  lag=self.features_lag,
                                  exclude_cols=self.features_lag_exclude_cols,
                                  verbose=self.verbose)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_rolling_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add rolling features because there are no original features.")
        _out_df = self._rolling_features(df=work_df,
                                         windows=self.features_rolling,
                                         exclude_cols=self.features_rolling_exclude_cols)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_additional_datacols(self) -> pd.DataFrame:

        # Dataframe that contains the target and all original features
        expanded_df = self.model_df.copy()

        # Add lagged cols
        if self.features_lag:
            expanded_df = self._create_lagged_variants(work_df=self.model_df[self.original_input_features].copy(),
                                                       expanded_df=expanded_df)

        if self.features_rolling:
            expanded_df = self._create_rolling_features(work_df=self.model_df[self.original_input_features].copy(),
                                                        expanded_df=expanded_df)

        if self.vectorize_timestamps:
            expanded_df = vectorize_timestamps(df=expanded_df, txt="")
            # For cyclical variables, keep only the sine/cosine variants, drop linear versions
            expanded_df = expanded_df.drop(columns=['.HOUR', '.SEASON', '.MONTH', '.WEEK', '.DOY'])

        if self.add_continuous_record_number:
            expanded_df = fr.add_continuous_record_number(df=expanded_df)

        # Timestamp sanitizer
        if self.sanitize_timestamp:
            verbose = True if self.verbose > 0 else False
            tss = TimestampSanitizer(data=expanded_df, output_middle_timestamp=True, verbose=verbose)
            expanded_df = tss.get()

        return expanded_df

    def _shap_importance(self, model, X, X_names) -> DataFrame:
        """
        Calculate SHAP-based feature importance.

        Uses TreeExplainer for tree-based models (XGBoost, RandomForest).
        Returns mean absolute SHAP values as feature importance.
        """

        # Create explainer and calculate SHAP values
        # Handle XGBoost base_score parameter format issue with monkey-patch
        # Some XGBoost/environment combinations return base_score as '[-4.121306E0]' which
        # float() cannot parse. We monkey-patch float() to handle this.
        _builtin_float = float

        def _patched_float(x):
            """float() that handles bracket-enclosed scientific notation like '[-4.121306E0]'"""
            if isinstance(x, str):
                x_stripped = x.strip('[]')
                if x_stripped != x:  # Only use patched version if brackets were removed
                    return _builtin_float(x_stripped)
            return _builtin_float(x)

        # Temporarily replace float in builtins
        import builtins
        original_float = builtins.float
        builtins.float = _patched_float

        try:
            explainer = shap.TreeExplainer(model)
        finally:
            # Always restore original float
            builtins.float = original_float

        shap_values = explainer.shap_values(X)

        # Handle case where shap_values is a list (for some model types)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP values as importance
        importance_values = np.abs(shap_values).mean(axis=0)

        # Calculate standard deviation for reference
        importance_std = np.abs(shap_values).std(axis=0)

        # Create DataFrame with feature importances
        fidf = pd.DataFrame({
            'SHAP_IMPORTANCE': importance_values,
            'SHAP_SD': importance_std
        }, index=X_names)

        fidf = fidf.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        return fidf

    def _add_random_variable(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add random variable as benchmark for relevant feature importances
        random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
        df[random_col] = np.random.RandomState(self._random_state).randn(df.shape[0])
        return df, random_col

    # def _lag_features(self, features_lag_exclude_cols):
    #     """Add lagged variants of variables as new features"""
    #     exclude_cols = [self.target_col]
    #     if features_lag_exclude_cols:
    #         exclude_cols += features_lag_exclude_cols
    #     return diive.pkgs.createvar.laggedvariants.lagged_variants(df=self.model_df,
    #                                                                stepsize=self.features_lag_stepsize,
    #                                                                lag=self.features_lag,
    #                                                                exclude_cols=exclude_cols,
    #                                                                verbose=self.verbose)

    def _rolling_features(self, df: pd.DataFrame, windows: list, exclude_cols: list = None) -> pd.DataFrame:
        """Add rolling mean and std of feature columns at multiple window sizes.

        For each window size w and each feature column col (excluding target and
        any cols in exclude_cols), two new columns are added:
            '.{col}_mean{w}' — rolling mean over the previous w records
            '.{col}_std{w}'  — rolling std over the previous w records

        Rolling statistics use min_periods=1 so no new NaN values are introduced
        at the start of the series.

        Args:
            df: DataFrame with feature columns and DatetimeIndex.
            windows: List of window sizes in records (e.g. [6, 48] for 3h and 24h
                     at 30-min resolution).
            exclude_cols: Column names to skip. Target column is always excluded.

        Returns:
            DataFrame with additional rolling feature columns appended.
        """
        exclude = [self.target_col] + (exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude]
        newcols = []

        for w in windows:
            rolled = df[feature_cols].rolling(window=w, min_periods=1)
            mean_df = rolled.mean()
            std_df = rolled.std(ddof=0)  # population std to avoid NaN for window=1

            mean_df.columns = [f'.{c}_MEAN{w}' for c in feature_cols]
            std_df.columns = [f'.{c}_SD{w}' for c in feature_cols]

            df = pd.concat([df, mean_df, std_df], axis=1)
            newcols += mean_df.columns.tolist() + std_df.columns.tolist()

        if self.verbose:
            print(f"++ Added rolling features (windows={windows}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

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

        # Calculate SHAP-based feature importance and store in dataframe
        print(f">>> Using final model on all data to calculate SHAP feature importance ...")
        self._feature_importances = self._shap_importance(
            model=self._model, X=X, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (SHAP) ...")
            plot_feature_importance(feature_importances=self.feature_importances_)

        # Scores, using all targets
        print(f">>> Calculating prediction scores based on all data predicting {self.target_col} ...")
        self._scores = prediction_scores(predictions=pred_y, targets=y)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values based on all data ...")
            plot_observed_predicted(predictions=pred_y,
                                    targets=y,
                                    scores=self.scores_,
                                    infotxt=f"trained on training set, tested on FULL set",
                                    random_state=self._random_state)

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

        print(f"\nGap-filling {_num_still_missing} remaining missing records in "
              f"{self.target_gapfilled_col} using fallback model ...")

        if _num_still_missing > 0:

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
            print(f">>> Fallback model not necessary, all gaps were already filled.")
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
        gf_fallback_df = vectorize_timestamps(df=gf_fallback_df, txt="(ONLY FALLBACK)")

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


def plot_feature_importance(feature_importances: pd.DataFrame):
    """
    Plot SHAP feature importance as horizontal bar chart with error bars.

    Visualizes relative importance of features in the model with standard deviation
    as error bars. Features are sorted by importance for easy interpretation.
    """
    # Scientific color palette
    COLOR_BAR = '#003A70'  # Deep Blue
    COLOR_ERROR = '#C41E3A'  # Crimson Red (error bars)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, ax = plt.subplots(figsize=(10, max(8, len(feature_importances) * 0.35)), dpi=100)

    # Prepare data
    _fidf = feature_importances.copy().sort_values(by='SHAP_IMPORTANCE', ascending=True)
    importances = _fidf['SHAP_IMPORTANCE']
    errors = _fidf['SHAP_SD']
    labels = _fidf.index

    # Create horizontal bar chart
    bars = ax.barh(range(len(importances)), importances,
                   color=COLOR_BAR, alpha=0.85, edgecolor=COLOR_TEXT, linewidth=0.8)

    # Add error bars
    ax.errorbar(importances, range(len(importances)), xerr=errors,
                fmt='none', ecolor=COLOR_ERROR, elinewidth=2, capsize=4,
                capthick=2, alpha=0.8, zorder=3)

    # Styling
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color=COLOR_TEXT)
    ax.set_xlabel('Feature Importance (mean |SHAP value|)', fontsize=14,
                  color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Feature', fontsize=14, color=COLOR_TEXT, fontweight='600')
    ax.set_title('SHAP Feature Importance', fontsize=16, fontweight='bold',
                 color='black', pad=15)

    # Grid
    ax.grid(True, axis='x', alpha=0.4, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11,
                   length=4, width=1, color=COLOR_TEXT)
    ax.tick_params(axis='x', labelsize=10)

    # Add value labels on bars
    for i, (imp, err) in enumerate(zip(importances, errors)):
        ax.text(imp + err * 0.5, i, f'{imp:.3f}', va='center', fontsize=9.5,
                color=COLOR_TEXT, fontweight='500')

    fig.tight_layout()
    fig.show()


def plot_observed_predicted(targets: np.ndarray,
                            predictions: np.ndarray,
                            scores: dict,
                            infotxt: str = "",
                            random_state: int = None):
    """
    Plot observed vs. predicted values with enhanced visual styling.

    Creates a 2-panel figure showing:
    - Left: Actual vs. Predicted scatter with accuracy bands and perfect prediction line
    - Right: Residuals vs. Predicted with zero line and error regions

    Visual styling follows diive's Material Design theme with color-coded accuracy zones.
    """
    # Scientific color palette - high contrast, publication-ready
    COLOR_SCATTER = '#003A70'  # Deep Blue
    COLOR_RESIDUAL = '#C41E3A'  # Crimson Red
    COLOR_PERFECT = '#2C3E50'  # Dark Slate Blue-Gray
    COLOR_GOOD = '#F4A300'  # Golden Yellow (±10% error)
    COLOR_WARN = '#E67F0D'  # Deep Orange (±20% error)
    COLOR_ERROR = '#C41E3A'  # Crimson Red (>20% error)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_ZERO = '#000000'  # Black
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5.5), dpi=100)

    # ==================== PANEL 1: Actual vs. Predicted ====================
    ax = axs[0]

    # Calculate data ranges for reference lines and zones
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    margin = (max_val - min_val) * 0.05
    plot_min, plot_max = min_val - margin, max_val + margin

    # Add accuracy zones (filled regions)
    # Perfect prediction line (y=x) with ±10% and ±20% bands
    x_ref = np.array([plot_min, plot_max])

    # ±20% error zone (strong orange, widest)
    ax.fill_between(x_ref, x_ref * 0.80, x_ref * 1.20,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±20% error band')

    # ±10% error zone (strong green, narrower)
    ax.fill_between(x_ref, x_ref * 0.90, x_ref * 1.10,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±10% error band')

    # Perfect prediction line (diagonal)
    ax.plot(x_ref, x_ref, '--', color=COLOR_PERFECT, lw=2, alpha=0.9,
            label='Perfect prediction', zorder=2)

    # Scatter plot with custom styling
    ax.scatter(targets, predictions,
               c=COLOR_SCATTER, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Predictions')

    # Formatting
    ax.set_xlabel('Observed values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Actual vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])
    ax.set_ylim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== PANEL 2: Residuals vs. Predicted ====================
    ax = axs[1]
    residuals = targets - predictions

    # Calculate residual statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    # Add reference bands (±1σ, ±2σ)
    zero_line_y = [plot_min, plot_max]
    ax.fill_between([plot_min, plot_max], -2 * std_residual, 2 * std_residual,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±2σ region')
    ax.fill_between([plot_min, plot_max], -1 * std_residual, 1 * std_residual,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±1σ region')

    # Zero line (perfect predictions have zero residuals)
    ax.axhline(y=0, color=COLOR_ZERO, linestyle='-', linewidth=1.5, alpha=0.85, zorder=2)

    # Scatter plot for residuals
    ax.scatter(predictions, residuals,
               c=COLOR_RESIDUAL, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Residuals')

    # Formatting
    ax.set_xlabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Residuals (Observed − Predicted)', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Residuals vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== Figure Title and Info Box ====================
    n_vals = len(predictions)
    mae = scores['mae']
    rmse = scores['rmse']
    r2 = scores['r2']

    # Main title
    title_text = f"Model Predictions: {infotxt}" if infotxt else "Model Predictions"
    fig.suptitle(title_text, fontsize=18, fontweight='bold', color='black', y=0.99)

    # Info box with metrics (positioned in figure space)
    info_lines = [
        f"n = {n_vals:,} samples",
        f"MAE = {mae:.4f}",
        f"RMSE = {rmse:.4f}",
        f"R² = {r2:.4f}"
    ]
    info_text = '\n'.join(info_lines)

    fig.text(0.99, 0.01, info_text,
             fontsize=11, color=COLOR_TEXT, verticalalignment='bottom',
             horizontalalignment='right', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                       edgecolor=COLOR_GRID, linewidth=1.2, alpha=0.95))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


def plot_prediction_residuals_error_regr(model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         infotxt: str):
    """
    Plot residuals and prediction error diagnostics.

    Creates two diagnostic plots using yellowbrick visualizers:
    - Left: Q-Q plot of residuals to assess normality assumption
    - Right: Actual vs. Predicted with prediction error visualization

    Args:
        model: Fitted regression model with predict() method
        X_train: predictors in training data (n_samples, n_features)
        y_train: targets in training data (n_samples,)
        X_test: predictors in test data (n_samples, n_features)
        y_test: targets in test data (n_samples,)
        infotxt: text displayed in figure header for context

    Notes:
        - Q-Q plot: If residuals are normally distributed, points should fall on
          the diagonal line. Deviations suggest non-normality.
        - Prediction Error: Shows scatter of predictions vs. actual values with
          perfect prediction line (y=x) and prediction confidence region.

    References:
        - https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
        - https://www.scikit-yb.org/en/latest/api/regressor/peplot.html
    """

    # Scientific color palette for consistency
    COLOR_POINTS = '#003A70'  # Deep Blue
    COLOR_LINE = '#F4A300'  # Golden Yellow
    COLOR_ERROR = '#C41E3A'  # Crimson Red
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    # ==================== PLOT 1: Q-Q Plot (Residuals Normality) ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Residuals Analysis: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    # Q-Q Plot detects normality violations
    vis = ResidualsPlot(model, hist=False, qqplot=True, ax=ax)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Q-Q Plot (Normality Assessment)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Sample Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        else:
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)

    plt.tight_layout()
    vis.show()

    # ==================== PLOT 2: Prediction Error Plot ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Prediction Error: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    vis = PredictionError(model)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Actual vs. Predicted (with Confidence Region)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Actual Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update scatter point colors
    for collection in ax.collections:
        collection.set_edgecolor(COLOR_TEXT)
        collection.set_facecolor(COLOR_POINTS)
        collection.set_alpha(0.7)
        collection.set_linewidth(0.5)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o', 's']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        elif line.get_linestyle() == '--':
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)
            line.set_alpha(0.85)
        else:
            line.set_color(COLOR_ERROR)
            line.set_linewidth(1.5)
            line.set_alpha(0.7)

    plt.tight_layout()
    vis.show()
