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
from diive.core.ml.common import feature_importances, prediction_scores_regr, plot_prediction_residuals_error_regr
from diive.core.times.times import include_timestamp_as_cols
from diive.core.times.times import TimestampSanitizer

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
                        'criterion': ['squared_error'],
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
        self._scores = prediction_scores_regr(predictions=grid_predictions,
                                              targets=y_test,
                                              infotxt=f"trained on training set, "
                                                      f"tested on test set",
                                              showplot=True)


class RandomForestTS:

    def __init__(
            self,
            input_df: DataFrame,
            target_col: str or tuple,
            verbose: int = 0,
            perm_n_repeats: int = 10,
            test_size: float = 0.25,
            features_lag: list = None,
            features_lagmax: int = None,
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

            verbose:
                If *1*, more output is printed.

        Attributes:
            gapfilled_df
            - .PREDICTIONS_FULLMODEL uses the output from the full RF model where
              all features where available.
            - .PREDICTIONS_FALLBACK uses the output from the fallback RF model, which
              was trained on the combined observed + .PREDICTIONS_FULLMODEL data, using
              only the timestamp info as features.
        """
        self.model_df = input_df.copy()
        self.target_col = target_col
        self.verbose = verbose
        self.kwargs = kwargs
        self.perm_n_repeats = perm_n_repeats
        self.test_size = test_size
        self.features_lag = features_lag
        self.features_lagmax = features_lagmax

        self._check_n_cols()

        if self.features_lag:
            self.model_df = self._lag_features()

        if include_timestamp_as_features:
            self.model_df = include_timestamp_as_cols(df=self.model_df, txt="")

        if add_continuous_record_number:
            self.model_df = fr.add_continuous_record_number(df=self.model_df)

        if sanitize_timestamp:
            tss = TimestampSanitizer(data=self.model_df, output_middle_timestamp=True, verbose=True)
            self.model_df = tss.get()

        # Attributes
        self._gapfilling_df = None  # Will contain gapfilled target and auxiliary variables
        self._model = None
        self._feature_importances = dict()
        self._scores = dict()
        self._feature_importances_test = dict()
        self._scores_test = dict()
        self._traintest_details = dict()

    def _lag_features(self):
        return fr.lagged_variants(df=self.model_df,
                                  stepsize=1,
                                  lag=self.features_lag,
                                  exclude_cols=[self.target_col])

    def _check_n_cols(self):
        """Check number of columns"""
        if len(self.model_df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.model_df.columns}")

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df[self.target_gapfilled_col]

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1=gap-filled, 2=gap-filled with fallback"""
        return self.gapfilling_df[self.target_gapfilled_flag_col]

    @property
    def model(self) -> RandomForestRegressor:
        """Return model, trained on test data"""
        if not self._model:
            raise Exception(f'Not available: model.')
        return self._model

    @property
    def feature_importances(self) -> dict:
        """Return feature importance for model used in gap-filling"""
        if not self._feature_importances:
            raise Exception(f'Not available: feature importances for gap-filling.')
        return self._feature_importances

    @property
    def scores(self) -> dict:
        """Return model scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    @property
    def gapfilling_df(self) -> DataFrame:
        """Return gapfilled data and auxiliary variables"""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception(f'Gapfilled data not available.')
        return self._gapfilling_df

    @property
    def feature_importances_test(self) -> dict:
        """Return feature importance from model training on training data,
        with importances calculated using test data (holdout set)"""
        if not self._feature_importances_test:
            raise Exception(f'Not available: feature importances for gap-filling.')
        return self._feature_importances_test

    @property
    def scores_test(self) -> dict:
        """Return model scores for model trained on training data,
        with scores calculated using test data (holdout set)"""
        if not self._scores_test:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores_test

    @property
    def traintest_details(self) -> dict:
        """Return details from train/test splits"""
        if not self._traintest_details:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._traintest_details

    def fillgaps(self,
                 showplot_scores: bool = True,
                 showplot_importance: bool = True,
                 verbose: int = 1
                 ):
        """
        Gap-fill data with previously built model

        No new model is built here, instead the last model built in
        the preceding step .trainmodel() is used.

        y = target
        X = features

        """
        self._fillgaps_fullmodel(showplot_scores, showplot_importance, verbose)
        self._fillgaps_fallback()
        self._fillgaps_combinepredictions()

    def report(self):

        idtxt = f"({self.report.__name__} gap-filling) "

        # Gapfilling stats from flag
        locs_observed = self.gapfilling_df[self.target_gapfilled_flag_col] == 0
        locs_observed_missing_fromflag = self.gapfilling_df[self.target_gapfilled_flag_col] > 0
        locs_observed_missing_fromdata = self.gapfilling_df[self.target_col].isnull()
        locs_gapfilled_missing = self.gapfilling_df[self.target_gapfilled_col].isnull()
        locs_hq = self.gapfilling_df[self.target_gapfilled_flag_col] == 1
        locs_fallback = self.gapfilling_df[self.target_gapfilled_flag_col] == 2

        n_available = len(self.gapfilling_df[self.target_gapfilled_col].dropna())
        n_potential = len(self.gapfilling_df.index)
        n_observed = locs_observed.sum()
        n_observed_missing_fromflag = locs_observed_missing_fromflag.sum()
        n_observed_missing_fromdata = locs_observed_missing_fromdata.sum()
        n_gapfilled_missing = locs_gapfilled_missing.sum()
        n_hq = locs_hq.sum()
        n_fallback = locs_fallback.sum()

        feature_names = self.feature_importances['importances'].index.to_list()
        n_features = len(feature_names)

        print(
            f"{idtxt} GAP-FILLING RESULTS\n"
            f"{idtxt}   First timestamp:  {self.gapfilling_df.index[0]}\n"
            f"{idtxt}   Last timestamp:  {self.gapfilling_df.index[-1]}\n"
            f"{idtxt}   Potential number of values: {n_potential} values)\n"
            f"{idtxt}\n"
            f"{idtxt}   TARGET\n"
            f"{idtxt}   - target column (observed):  {self.target_col}\n"
            f"{idtxt}   - missing records (observed):  {n_observed_missing_fromdata} "
            f"(cross-check from flag: {n_observed_missing_fromflag})\n"
            f"{idtxt}   - target column (gap-filled):  {self.target_gapfilled_col}  ({n_available} values)\n"
            f"{idtxt}   - missing records (gap-filled):  {n_gapfilled_missing}\n"
            f"{idtxt}   - gap-filling flag: {self.target_gapfilled_flag_col}\n"
            f"{idtxt}       - flag 0 ... observed targets ({n_observed} values)\n"
            f"{idtxt}       - flag 1 ... targets gap-filled with high-quality, all features available "
            f"({n_hq} values)\n"
            f"{idtxt}       - flag 2 ... targets gap-filled with fallback ({n_fallback} values)\n"
            f"{idtxt}\n"
            f"{idtxt}   MODEL\n"
            f"{idtxt}   The model was trained on a training set.\n"
            f"{idtxt}   - estimator:  {self.model}\n"
            f"{idtxt}   - parameters:  {self.model.get_params()}\n"
            f"{idtxt}   - names of features used in model:  {feature_names}\n"
            f"{idtxt}   - number of features used in model:  {n_features}\n"
            f"{idtxt}\n"
            f"{idtxt}   MODEL SCORES\n"
            f"{idtxt}   Model scores were calculated from high-quality predicted targets "
            f"({n_hq} values, {self.target_gapfilled_col} where flag=1) in comparison to "
            f"observed targets ({n_observed} values, {self.target_col}).\n"
            f"{idtxt}   - MAE:  {self.scores['mae']} (mean absolute error)\n"
            f"{idtxt}   - MedAE:  {self.scores['medae']} (median absolute error)\n"
            f"{idtxt}   - MSE:  {self.scores['mse']} (mean squared error)\n"
            f"{idtxt}   - RMSE:  {self.scores['rmse']} (root mean squared error)\n"
            f"{idtxt}   - MAXE:  {self.scores['maxe']} (max error)\n"
            f"{idtxt}   - MAPE:  {self.scores['mape']:.3f} (mean absolute percentage error)\n"
            f"{idtxt}   - R2:  {self.scores['r2']}\n"
            f"{idtxt}\n"
            f"{idtxt}   FEATURE IMPORTANCES\n"
            f"{idtxt}   Feature importances were calculated from high-quality predicted targets "
            f"({n_hq} values, {self.target_gapfilled_col} where flag=1) in comparison to "
            f"observed targets ({n_observed} values, {self.target_col}).\n"
            f"{idtxt}\n{self.feature_importances['importances']}\n"
            f"{idtxt}   Permutation importances were calculated from {self.perm_n_repeats} repeats.\n"
        )

    def _fillgaps_fullmodel(self, showplot_scores, showplot_importance, verbose):
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
        y, X, X_names, timestamp = \
            fr.convert_to_arrays(df=df,
                                 target_col=self.target_col,
                                 complete_rows=True)

        # Predict all targets (no test split)
        pred_y = self.model.predict(X=X)

        # Calculate feature importance, using all targets
        self._feature_importances = feature_importances(estimator=self.model,
                                                        X=X,
                                                        y=y,
                                                        model_feature_names=X_names,
                                                        random_col=None,
                                                        perm_n_repeats=self.perm_n_repeats,
                                                        showplot=showplot_importance)

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
        pred_y = self.model.predict(X=X)

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

    def trainmodel(self,
                   remove_features_low_importance: bool = True,
                   showplot_predictions: bool = True,
                   showplot_importance: bool = True,
                   verbose: int = 1):
        """
        Train random forest model for gap-filling

        No gap-filling is done here, only the model is trained.

        Args:
            remove_features_low_importance: If *True*, variables with lower permutation importance
                than a random variable are not included in the model training.
            showplot_predictions: shows plot of predicted vs observed
            showplot_importance: shows plot of permutation importances
            verbose: if > 0 prints more text output

        """

        if remove_features_low_importance:
            # Train model with random variable included, to detect unimportant features
            self._trainmodel(df=self.model_df.copy(),
                             include_random=True,
                             showplot_scores=showplot_predictions,
                             showplot_importance=showplot_importance,
                             verbose=verbose)

            recommended_features = self.feature_importances_test['recommended_features']
            usecols = [self.target_col]
            usecols = usecols + recommended_features
            self.model_df = self.model_df[usecols].copy()

        # Train model for gap-filling
        self._trainmodel(df=self.model_df,
                         include_random=False,
                         showplot_scores=showplot_predictions,
                         showplot_importance=showplot_importance,
                         verbose=verbose)

    def _trainmodel(self,
                    df: DataFrame,
                    include_random: bool = False,
                    showplot_scores: bool = True,
                    showplot_importance: bool = True,
                    verbose: int = 1):
        """Build and test model

        See:
            - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        """

        # Add random variable as benchmark for relevant feature importances
        if include_random:
            random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
            df[random_col] = np.random.RandomState(self.kwargs['random_state']).randn(df.shape[0], 1)
            # df[random_col] = np.random.rand(df.shape[0], 1)
            idtxt = f"({self._trainmodel.__name__} with random) "
        else:
            random_col = None
            idtxt = f"({self._trainmodel.__name__}) "

        # Info
        if verbose > 0:
            print(f"{idtxt}Building random forest model based on data between "
                  f"{df.index[0]} and {df.index[-1]} ...")

        # Data as arrays
        # y = targets, X = features
        y, X, X_names, timestamp = \
            fr.convert_to_arrays(df=df,
                                 target_col=self.target_col,
                                 complete_rows=True)

        # Train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.kwargs['random_state'])

        # Instantiate model with params
        self._model = RandomForestRegressor(**self.kwargs)

        # Train the model
        self._model.fit(X=X_train, y=y_train)

        # Predict targets in test data
        pred_y_test = self._model.predict(X=X_test)

        # Calculate feature importance
        self._feature_importances_test = feature_importances(estimator=self._model,
                                                             X=X_test,
                                                             y=y_test,
                                                             model_feature_names=X_names,
                                                             random_col=random_col,
                                                             perm_n_repeats=self.perm_n_repeats,
                                                             showplot=showplot_importance)

        # Stats
        self._scores_test = prediction_scores_regr(predictions=pred_y_test,
                                                   targets=y_test,
                                                   infotxt=f"{idtxt} trained on training set, "
                                                           f"tested on test set",
                                                   showplot=showplot_scores)

        if showplot_scores:
            plot_prediction_residuals_error_regr(model=self._model,
                                                 X_train=X_train,
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 infotxt=f"{idtxt} trained on training set, "
                                                         f"tested on test set")

        # Collect results
        self._traintest_details = dict(
            X=X,
            y=y,
            timestamp=timestamp,
            predictions=pred_y_test,
            include_random=include_random,
            X_names=X_names,
            y_name=self.target_col,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=self._model,
        )

        if self.verbose > 0:
            self.report_training(idtxt=idtxt, results=self.traintest_details)

    def report_training(self, idtxt: str, results: dict):
        """Results from model training on test data"""
        test_size_perc = self.test_size * 100
        training_size_perc = 100 - test_size_perc
        n_vals_observed = len(results['y'])
        n_vals_train = len(results['y_train'])
        n_vals_test = len(results['y_test'])
        timestamp = results['timestamp']
        used_features = results['X_names']
        model = results['model']

        print(
            f"{idtxt} MODEL TRAINING & TESTING RESULTS\n"
            f"{idtxt}   - the model was trained and tested based on data between "
            f"{timestamp[0]} and {timestamp[-1]}.\n"
            f"{idtxt}   - in total, {n_vals_observed} observed target values were available for training and testing\n"
            f"{idtxt}   - the dataset was split into training and test datasets\n"
            f"{idtxt}       - the training dataset comprised {n_vals_train} target values ({training_size_perc:.1f}%)\n"
            f"{idtxt}       - the test dataset comprised {n_vals_test} target values ({test_size_perc:.1f}%)\n"
            f"{idtxt}\n"
            f"{idtxt}   MODEL\n"
            f"{idtxt}   The model was trained on the training set.\n"
            f"{idtxt}   - estimator:  {model}\n"
            f"{idtxt}   - parameters:  {model.get_params()}\n"
            f"{idtxt}   - names of features used in model:  {used_features}\n"
            f"{idtxt}   - number of features used in model:  {len(used_features)}\n"
            f"{idtxt}\n"
            f"{idtxt}   MODEL SCORES\n"
            f"{idtxt}   Model was trained on training data ({n_vals_train} values).\n"
            f"{idtxt}   Model was tested on test data ({n_vals_test} values).\n"
            f"{idtxt}   All scores were calculated for test split.\n"
            f"{idtxt}   - MAE:  {self.scores_test['mae']} (mean absolute error)\n"
            f"{idtxt}   - MedAE:  {self.scores_test['medae']} (median absolute error)\n"
            f"{idtxt}   - MSE:  {self.scores_test['mse']} (mean squared error)\n"
            f"{idtxt}   - RMSE:  {self.scores_test['rmse']} (root mean squared error)\n"
            f"{idtxt}   - MAXE:  {self.scores_test['maxe']} (max error)\n"
            f"{idtxt}   - MAPE:  {self.scores_test['mape']:.3f} (mean absolute percentage error)\n"
            f"{idtxt}   - R2:  {self.scores_test['r2']}\n"
            f"{idtxt}\n"
            f"{idtxt}   FEATURE IMPORTANCES\n"
            f"{idtxt}   Feature importances were calculated for test data ({n_vals_test} target values).\n"
            f"{idtxt}\n{self.feature_importances_test['importances']}\n"
            f"{idtxt}   Permutation importances were calculated from {self.perm_n_repeats} repeats."
        )

        if results['include_random']:
            print(
                f"{idtxt}\n"
                f"{idtxt} --> {len(self.feature_importances_test['recommended_features'])} recommended features: "
                f"{self.feature_importances_test['recommended_features']} "
                f"(permutation importance larger than random)\n"
            )

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
        model_fallback = RandomForestRegressor(**self.kwargs)

        # Train the model on all available records ...
        model_fallback.fit(X=X_fallback, y=y_fallback)

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


class QuickFillRFTS:
    """
    Quick gap-filling using RandomForestTS with pre-defined parameters

    The purpose of this class is preliminary gap-filling e.g. for quick tests
    how gap-filled data could look like. It is not meant to be used for
    final gap-filling.
    """

    def __init__(self, df: DataFrame, target_col: str or tuple):
        self.df = df.copy()
        self.target_col = target_col
        self.rfts = None

    def fill(self):
        self.df = include_timestamp_as_cols(df=self.df.copy(), txt="(...)")
        self.rfts = RandomForestTS(input_df=self.df, target_col=self.target_col, n_estimators=20)
        self.rfts.trainmodel(showplot_predictions=False, showplot_importance=False, verbose=0)
        self.rfts.fillgaps(showplot_scores=True, showplot_importance=True, verbose=1)

    def gapfilling_df(self):
        return self.rfts.gapfilling_df

    def report(self):
        return self.rfts.report()

    def get_gapfilled(self) -> Series:
        return self.rfts.get_gapfilled_target()


def example_quickfill():
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
    gapfilled = qf.get_gapfilled()

    # Plot
    HeatmapDateTime(series=df[TARGET_COL]).show()
    HeatmapDateTime(series=gapfilled).show()


def example_rfts():
    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Setup, imports
    import numpy as np
    import importlib.metadata
    from datetime import datetime
    import matplotlib.pyplot as plt
    from diive.configs.exampledata import load_exampledata_parquet
    from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    from diive.core.dfun.stats import sstats  # Time series stats
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"This page was last modified on: {dt_string}")
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")

    # Show docstring for QuickFillRFTS
    print(RandomForestTS.__name__)
    print(RandomForestTS.__doc__)

    # Example data
    df = load_exampledata_parquet()

    # # Create a large gap
    # remove = df.index.year != 2014
    # # df = df.drop(df.index[100:2200])
    # df = df[remove].copy()

    # # Subset
    # keep = df.index.year <= 2016
    # df = df[keep].copy()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    lowquality = df["QCF_NEE"] > 0
    df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()
    df.describe()
    statsdf = sstats(df[TARGET_COL])
    print(statsdf)
    TimeSeries(series=df[TARGET_COL]).plot()

    # Random forest
    rfts = RandomForestTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=1,
        # features_lag=None,
        features_lag=[-1, -1],
        include_timestamp_as_features=True,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
        n_estimators=200,
        random_state=42,
        min_samples_split=10,
        min_samples_leaf=5,
        perm_n_repeats=11,
        n_jobs=-1
    )
    rfts.trainmodel(showplot_predictions=True, showplot_importance=True, verbose=1)
    rfts.fillgaps(showplot_scores=True, showplot_importance=True, verbose=1)
    rfts.report()
    observed = df[TARGET_COL]
    gapfilled = rfts.get_gapfilled_target()
    # rfts.feature_importances
    # rfts.scores
    # rfts.gapfilling_df

    # Plot
    HeatmapDateTime(series=observed).show()
    HeatmapDateTime(series=gapfilled).show()

    # rfts.gapfilling_df['.PREDICTIONS_FALLBACK'].cumsum().plot()
    # rfts.gapfilling_df['.PREDICTIONS_FULLMODEL'].cumsum().plot()
    # rfts.gapfilling_df['.PREDICTIONS'].cumsum().plot()
    rfts.gapfilling_df['NEE_CUT_REF_orig_gfRF'].cumsum().plot()
    plt.legend()
    plt.show()

    # d = rfts.gapfilling_df['NEE_CUT_REF_orig'] - rfts.gapfilling_df['.PREDICTIONS']
    # d.plot()
    # plt.show()
    # d = abs(d)
    # d.mean()  # MAE


    print("Finished.")


def example_optimize():
    from diive.configs.exampledata import load_exampledata_parquet

    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    df = load_exampledata_parquet()
    subset = df[subsetcols].copy()
    _subset = df.index.year == 2019
    subset = subset[_subset].copy()

    # Random forest parameters
    rf_params = {
        'n_estimators': list(range(2, 12, 2)),
        'criterion': ['squared_error'],
        'max_depth': [None],
        'min_samples_split': list(range(2, 12, 2)),
        'min_samples_leaf': list(range(1, 6, 1))
    }

    # Optimization
    opt = OptimizeParamsRFTS(
        df=subset,
        target_col=TARGET_COL,
        **rf_params
    )

    opt.optimize()

    print(opt.best_params)
    print(opt.scores)
    print(opt.cv_results)
    print(opt.best_score)
    print(opt.cv_n_splits)


if __name__ == '__main__':
    # example_quickfill()
    example_rfts()
    # example_optimize()
