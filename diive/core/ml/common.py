# todo check for other estimators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay, max_error, median_absolute_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score, mean_squared_error
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
            verbose: int = 0,
            perm_n_repeats: int = 10,
            test_size: float = 0.25,
            features_lag: list = None,
            include_timestamp_as_features: bool = False,
            add_continuous_record_number: bool = False,
            sanitize_timestamp: bool = False,
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
        self.model_df = input_df.copy()
        self.target_col = target_col
        self.kwargs = kwargs
        self.perm_n_repeats = perm_n_repeats
        self.test_size = test_size
        self.features_lag = features_lag
        self.verbose = verbose

        if self.features_lag and (len(self.model_df.columns) > 1):
            self.model_df = self._lag_features()

        if include_timestamp_as_features:
            self.model_df = include_timestamp_as_cols(df=self.model_df, txt="")

        if add_continuous_record_number:
            self.model_df = fr.add_continuous_record_number(df=self.model_df)

        if sanitize_timestamp:
            verbose = True if verbose > 0 else False
            tss = TimestampSanitizer(data=self.model_df, output_middle_timestamp=True, verbose=verbose)
            self.model_df = tss.get()

        self._check_n_cols()

        self.random_col = None

        # Instantiate model with params
        self._model = self.regressor(**self.kwargs)

        # Attributes
        self._gapfilling_df = None  # Will contain gapfilled target and auxiliary variables
        # self._model = None
        self._feature_importances = dict()
        self._feature_importances_traintest = dict()
        self._feature_importances_reduction = dict()
        self._scores = dict()
        self._scores_test = dict()
        self._accepted_features = []
        self._rejected_features = []

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
    def feature_importances_reduction_(self) -> DataFrame:
        """Return feature importance from feature reduction, model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_reduction, DataFrame):
            raise Exception(f'Not available: feature importances from feature reduction.')
        return self._feature_importances_reduction

    @property
    def scores_(self) -> dict:
        """Return model scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    @property
    def scores_test_(self) -> dict:
        """Return model scores for model trained on training data,
        with scores calculated using test data (holdout set)"""
        if not self._scores_test:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores_test

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

    def reduce_features(self):
        """Reduce number of features using permutation importance

        A random variable is added to features and the permutation importances
        are calculated. The permutation importance of the random variable is the
        benchmark to determine whether a feature is relevant. All features where
        permutation importance is smaller or equal to the importance of the random
        variable are rejected.
        """

        df = self.model_df.copy()

        # Info
        print(f"\n\nFeature reduction based on permutation importance ...")

        # Add random variable as feature
        df, self.random_col = self._add_random_variable(df=df)

        # Data as arrays, y = targets, X = features
        y, X, X_names, timestamp = fr.convert_to_arrays(
            df=df, target_col=self.target_col, complete_rows=True)

        # Train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.kwargs['random_state'])

        # Instantiate model with params
        model = self.regressor(**self.kwargs)

        # Fit model
        model = self._fitmodel(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # # Predict targets in test data
        # pred_y_test = model.predict(X=X_test)

        # Calculate permutation importance and store in dataframe
        self._feature_importances_reduction = self._permutation_importance(
            model=model, X=X_test, y=y_test, X_names=X_names, showplot_importance=False)

        # Update dataframe for model building
        self.model_df = self._remove_rejected_features(df=df.copy())

        # # This could be a way to combine permutation importance with RFECV,
        # # but at the time of this writing an import failed (Oct 2023)
        # # Train model with random variable included, to detect unimportant features
        # df = df.dropna()
        # targets = df[self.target_col].copy()
        # df = df.drop(self.target_col, axis=1, inplace=False)
        # features = df.copy()
        # estimator = self.regressor(**self.kwargs)
        # splitter = TimeSeriesSplit(n_splits=10)
        # from eli5.sklearn import PermutationImportance
        # rfecv = RFECV(estimator=PermutationImportance(estimator, scoring='r2', n_iter=10, random_state=42, cv=splitter),
        #               step=1,
        #               min_features_to_select=3,
        #               cv=splitter,
        #               scoring='r2',
        #               verbose=self.verbose,
        #               n_jobs=-1)
        # rfecv.fit(features, targets)
        # # Feature importances
        # features.drop(features.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
        # rfecv_fi_df = pd.DataFrame()
        # rfecv_fi_df['FEATURE'] = list(features.columns)
        # rfecv_fi_df['IMPORTANCE'] = rfecv.estimator_.feature_importances_
        # rfecv_fi_df = rfecv_fi_df.set_index('FEATURE')
        # rfecv_fi_df = rfecv_fi_df.sort_values(by='IMPORTANCE', ascending=False)
        # # rfecv.cv_results_
        # # rfecv.n_features_
        # # rfecv.n_features_in_
        # # rfecv.ranking_
        # # rfecv.support_

    def _remove_rejected_features(self, df: pd.DataFrame):
        """Remove features below importance threshold from model dataframe.
        The updated model dataframe will then be used for the next (final) model.
        """
        # Threshold for feature acceptance
        fi_threshold = self.feature_importances_reduction_['PERM_IMPORTANCE'][self.random_col]

        # Get accepted and rejected features
        fidf_accepted = self.feature_importances_reduction_.loc[
            self.feature_importances_reduction_['PERM_IMPORTANCE'] > fi_threshold].copy()
        self._accepted_features = fidf_accepted.index.tolist()
        fidf_rejected = self.feature_importances_reduction_.loc[
            self.feature_importances_reduction_['PERM_IMPORTANCE'] <= fi_threshold].copy()
        self._rejected_features = fidf_rejected.index.tolist()

        # Assemble dataframe for next model
        usecols = [self.target_col]
        usecols = usecols + self._accepted_features
        df = df[usecols].copy()
        return df

    @staticmethod
    def _fitmodel(model, X_train, y_train, X_test, y_test):
        """Fit model."""
        print(f"Fitting model {type(model)} ...")
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
            showplot_predictions: shows plot of predicted vs observed
            showplot_importance: shows plot of permutation importances
            verbose: if > 0 prints more text output

        """

        df = self.model_df.copy()

        # Info
        idtxt = f"TRAIN & TEST "
        print(f"Building {type(self._model)} model based on data between "
              f"{df.index[0]} and {df.index[-1]} ...")

        # Data as arrays
        # y = targets, X = features
        y, X, X_names, timestamp = fr.convert_to_arrays(
            df=df, target_col=self.target_col, complete_rows=True)

        # Train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.kwargs['random_state'])

        # Train the model
        self._model = self._fitmodel(model=self._model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        # self._model.fit(X=X_train, y=y_train)

        # Predict targets in test data
        pred_y_test = self._model.predict(X=X_test)

        # Calculate permutation importance and store in dataframe
        self._feature_importances_traintest = self._permutation_importance(
            model=self._model, X=X_test, y=y_test, X_names=X_names, showplot_importance=showplot_importance)

        # Stats
        self._scores_test = prediction_scores_regr(
            predictions=pred_y_test, targets=y_test, showplot=showplot_scores,
            infotxt=f"{idtxt} trained on training set, tested on test set")

        if showplot_scores:
            plot_prediction_residuals_error_regr(
                model=self._model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                infotxt=f"{idtxt} trained on training set, tested on test set")

        # Collect results
        self._traintest_details = dict(
            X=X,
            y=y,
            timestamp=timestamp,
            predictions=pred_y_test,
            X_names=X_names,
            y_name=self.target_col,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=self._model,
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

    def report_feature_reduction(self):
        """Results from feature reduction"""

        idtxt = "FEATURE REDUCTION"

        fi = self.feature_importances_reduction_

        # TODO hier weiter

        _X_names = [x for x in fi.index if x != self.random_col]  # Original features without random variable
        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"- the random variable {self.random_col} was added to the original features, "
            f"used as benchmark for detecting relevant feature importances\n"
            f"- target variable: {self.target_col}\n"
            f"- features before reduction: {fi.index.to_list()}\n"
            f"- permutation importance was calculated from {self.perm_n_repeats} permutations\n"
            f"- These results are from feature reduction. Note that feature importances for "
            f"the final model are calculated during gap-filling.\n"
            f"\n"
            f"\n"
            f"PERMUTATION IMPORTANCE (FULL RESULTS):\n"
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"--> {len(fi.index)} input features, "
            f"including {self.random_col}: {fi.index.tolist()}\n"
            f"--> {len(self.accepted_features_)} accepted features, "
            f"larger than {self.random_col}: {self.accepted_features_}\n"
            f"--> {len(self.rejected_features_)} rejected features, "
            f"smaller than or equal to {self.random_col}: {self.rejected_features_}\n"
        )

    def report_traintest(self):
        """Results from model training on test data"""

        idtxt = "MODEL TRAINING & TESTING RESULTS"

        results = self.traintest_details_
        fi = self.feature_importances_traintest_

        test_size_perc = self.test_size * 100
        training_size_perc = 100 - test_size_perc
        n_vals_observed = len(results['y'])
        n_vals_train = len(results['y_train'])
        n_vals_test = len(results['y_test'])
        timestamp = results['timestamp']
        used_features = results['X_names']
        model = results['model']

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"- the model was trained and tested based on data between "
            f"{timestamp[0]} and {timestamp[-1]}.\n"
            f"- in total, {n_vals_observed} observed target values were available for training and testing\n"
            f"- the dataset was split into training and test datasets\n"
            f"  > the training dataset comprised {n_vals_train} target values ({training_size_perc:.1f}%)\n"
            f"  > the test dataset comprised {n_vals_test} target values ({test_size_perc:.1f}%)\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"- feature importances were calculated for test data ({n_vals_test} target values).\n"
            f"- permutation importances were calculated from {self.perm_n_repeats} repeats."
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"## MODEL\n"
            f"The model was trained on the training set.\n"
            f"- estimator:  {model}\n"
            f"- parameters:  {model.get_params()}\n"
            f"- names of features used in model:  {used_features}\n"
            f"- number of features used in model:  {len(used_features)}\n"
            f"\n"
            f"## MODEL SCORES\n"
            f"- the model was trained on training data ({n_vals_train} values).\n"
            f"- the model was tested on test data ({n_vals_test} values).\n"
            f"- all scores were calculated for test split.\n"
            f"  > MAE:  {self.scores_test_['mae']} (mean absolute error)\n"
            f"  > MedAE:  {self.scores_test_['medae']} (median absolute error)\n"
            f"  > MSE:  {self.scores_test_['mse']} (mean squared error)\n"
            f"  > RMSE:  {self.scores_test_['rmse']} (root mean squared error)\n"
            f"  > MAXE:  {self.scores_test_['maxe']} (max error)\n"
            f"  > MAPE:  {self.scores_test_['mape']:.3f} (mean absolute percentage error)\n"
            f"  > R2:  {self.scores_test_['r2']}\n"
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

    def _permutation_importance(self, model, X, y, X_names, showplot_importance) -> DataFrame:
        """Calculate permutation importance"""

        print(f"Calculating permutation importance using model {type(model)} ...")

        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-feature-importance
        fi = permutation_importance(estimator=model,
                                    X=X, y=y,
                                    n_repeats=self.perm_n_repeats,
                                    random_state=42,
                                    scoring='r2',
                                    n_jobs=-1)

        # Store permutation importance
        fidf = pd.DataFrame({'PERM_IMPORTANCE': fi.importances_mean,
                             'PERM_SD': fi.importances_std},
                            index=X_names)

        fidf = fidf.sort_values(by='PERM_IMPORTANCE', ascending=False)

        if showplot_importance:
            fig, axs = plt.subplots(ncols=1, figsize=(9, 16))
            _fidf = fidf.copy().sort_values(by='PERM_IMPORTANCE', ascending=True)
            _fidf['PERM_IMPORTANCE'].plot.barh(color='#008bfb', yerr=_fidf['PERM_SD'], ax=axs)
            axs.set_xlabel("Feature importance")
            axs.set_ylabel("Feature")
            axs.set_title(f"Permutation importance ({self.perm_n_repeats} permutations)")
            axs.legend(loc='lower right')
            fig.tight_layout()
            fig.show()

        return fidf

    def _add_random_variable(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add random variable as benchmark for relevant feature importances
        random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
        df[random_col] = np.random.RandomState(self.kwargs['random_state']).randn(df.shape[0], 1)
        # df[random_col] = np.random.rand(df.shape[0], 1)
        return df, random_col

    def _lag_features(self):
        """Add lagged variants of variables as new features"""
        return fr.lagged_variants(df=self.model_df,
                                  stepsize=1,
                                  lag=self.features_lag,
                                  exclude_cols=[self.target_col])

    def _check_n_cols(self):
        """Check number of columns"""
        if len(self.model_df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.model_df.columns}")

    def _fillgaps_fullmodel(self, showplot_scores, showplot_importance):
        """Apply model to fill missing targets for records where all features are available
        (high-quality gap-filling)"""

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
        pred_y = self.model_.predict(X=X)

        # Calculate permutation importance and store in dataframe
        self._feature_importances = self._permutation_importance(
            model=self._model, X=X, y=y, X_names=X_names, showplot_importance=showplot_importance)

        # Model scores, using all targets
        self._scores = prediction_scores_regr(predictions=pred_y,
                                              targets=y,
                                              infotxt="trained on training set, "
                                                      "tested on full set",
                                              showplot=showplot_scores)

        # In the next step, all available features are used to
        # predict the target for records where all features are available.
        # Feature data for records where all features are available:
        features_df = df.drop(self.target_col, axis=1)  # Remove target data
        features_df = features_df.dropna()  # Keep rows where all features available
        X = features_df.to_numpy()  # Features are needed as numpy array
        feature_names = features_df.columns.tolist()

        # Predict targets for all records where all features are available
        pred_y = self.model_.predict(X=X)

        # Collect gapfilling results in df
        # Define column names for gapfilled_df
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
        self._gapfilling_df[self.target_gapfilled_col] = \
            self._gapfilling_df[self.target_col].fillna(self._gapfilling_df[self.pred_fullmodel_col])

        # Restore original full timestamp
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
        model_fallback = self._fitmodel(model=model_fallback, X_train=X_fallback, y_train=y_fallback, X_test=X_fallback, y_test=y_fallback)
        # model_fallback.fit(X=X_fallback, y=y_fallback)

        # ... and use it to predict all records for full timestamp
        full_timestamp_df = gf_fallback_df.drop(self.target_gapfilled_col, axis=1)  # Remove target data
        X_fallback_full = full_timestamp_df.to_numpy()  # Features are needed as numpy array
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
        self.target_gapfilled_col = f"{self.target_col}_gfRF"
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
#     perm_results = permutation_importance(estimator, X, y, n_repeats=perm_n_repeats, random_state=42,
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
                           targets: np.array,
                           infotxt: str = None,
                           showplot: bool = True,
                           verbose: int = 0) -> dict:
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
        'rmse': mean_squared_error(targets, predictions, squared=False),  # root mean squared error
        'mape': mean_absolute_percentage_error(targets, predictions),
        'maxe': max_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }

    # Plot observed and predicted
    if showplot:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            targets,
            y_pred=predictions,
            kind="actual_vs_predicted",
            subsample=None,
            ax=axs[0],
            random_state=42,
        )
        axs[0].set_title("Actual vs. Predicted values")
        PredictionErrorDisplay.from_predictions(
            targets,
            y_pred=predictions,
            kind="residual_vs_predicted",
            subsample=None,
            ax=axs[1],
            random_state=42,
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        n_vals = len(predictions)
        fig.suptitle(f"Plotting cross-validated predictions ({infotxt})\n"
                     f"n_vals={n_vals}, MAE={scores['mae']:.3f}, RMSE={scores['rmse']:.3f}, r2={scores['r2']:.3f}")
        plt.tight_layout()
        plt.show()

    return scores


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
