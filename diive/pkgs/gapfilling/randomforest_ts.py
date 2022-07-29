"""
=========================================
RANDOM FOREST GAP-FILLING FOR TIME SERIES
randomforest_ts
=========================================

This module is part of the diive library:
https://gitlab.ethz.ch/diive/diive



"""
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.dfun.frames as frames
from diive.core.ml.common import train_random_forest_regressor, model_importances, mape_acc
from diive.core.times.times import include_timestamp_as_cols

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class RandomForestTS:

    def __init__(
            self,
            df: DataFrame,
            target_col: str or tuple,
            # feature_cols:list,
            n_estimators: int = 100,
            criterion: str = 'squared_error',
            max_depth: int = None,
            min_samples_split: int or float = 2,
            min_samples_leaf: int or float = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: int or float = 'auto',
            max_leaf_nodes: int = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: bool = False,
            n_jobs: int = -1,  # Custom default
            random_state: int = None,
            verbose: int = 0,
            warm_start: bool = False,
            ccp_alpha: float = 0.0,  # Non-negative
            max_samples: int or float = None
    ):
        """

        Parameters
        ----------
        df: pd.Dataframe
            Contains data and timestamp (in index) for target and features.
        target_col: str or tuple
            Column name of variable that will be gap-filled.
        lagged_variants: int, default=None
            Create lagged variants of non-target columns, shift by value.
        threshold_important_features: float
            Threshold for feature importance. Variables with importances
            below this threshold are removed from the final model.
        week_as_feature
        month_as_feature
        doy_as_feature
        n_estimators
        criterion
        max_depth
        min_samples_split
        min_samples_leaf
        min_weight_fraction_leaf
        max_features
        max_leaf_nodes
        min_impurity_decrease
        min_impurity_split
        bootstrap
        oob_score
        n_jobs
        random_state
        verbose
        warm_start
        ccp_alpha
        max_samples
        """
        self.df = df
        self.target_col = target_col
        # self.feature_cols=feature_cols
        self.random_state = random_state
        self.verbose = verbose
        self.n_estimators = n_estimators

        self.model_params = dict(
            random_state=self.random_state,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )

        # From .build_final_model
        self.model_results = dict()
        self.model = None

        # From .gapfilling
        self.gapfilled_df = None
        self.gf_results = None

    def run(self):
        # todo traintestsplit?

        # Prepare data
        self.df = self._include_timestamp_as_features()

        # Build model for gapfilling
        self.model_results = self._build_model()

        print(self.model_results['mae'])
        print(self.model_results['r2'])

        self.model = self.model_results['model']

        # Gapfilling
        self.gapfilled_df, self.gf_results = self._fill()

    def _include_timestamp_as_features(self, doy_as_feature: bool = True, hour_as_feature: bool = True,
                                       week_as_feature: bool = True, month_as_feature: bool = True) -> DataFrame:
        return include_timestamp_as_cols(df=self.df,
                                         doy=doy_as_feature,
                                         hour=hour_as_feature,
                                         week=week_as_feature,
                                         month=month_as_feature,
                                         info=True)

    def get_gapfilled_dataset(self):
        """

        Returns
        -------
        pd.Dataframe, dict
        """
        return self.gapfilled_df, self.gf_results

    def _build_model(self) -> dict:
        """Build final model for gapfilling"""

        _id_method = "[FINAL MODEL]    "

        if self.verbose > 0:
            print(f"\n\n{_id_method}START {'=' * 30}")

        # Data as arrays
        model_targets, model_features, model_feature_names, model_timestamp = \
            frames.convert_to_arrays(df=self.df,
                                     target_col=self.target_col,
                                     complete_rows=True)

        # Model training
        model, model_r2 = \
            train_random_forest_regressor(targets=model_targets,
                                          features=model_features,
                                          n_estimators=self.n_estimators,
                                          model_params=self.model_params)  # All features, all targets

        # from sklearn.inspection import permutation_importance
        # perm_importance = permutation_importance(model, model_features, model_targets)

        # Predict targets w/ features
        model_predictions = \
            model.predict(X=model_features)

        # from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        # # np.mean(cross_val_score(clf, X_train, y_train, cv=10))
        # tscv = TimeSeriesSplit(n_splits=5)
        # cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')

        # Stats
        model_mape, model_accuracy, model_mae = \
            mape_acc(predictions=model_predictions,
                     targets=model_targets)

        # Importances
        model_fi_all, \
        model_fi_most_important_df, \
        model_fi_most_important_list = \
            model_importances(model=model,
                              feature_names=model_feature_names,
                              threshold_important_features=None)

        model_results = dict(targets=model_targets,
                             features=model_features,
                             feature_names=model_feature_names,
                             timestamp=model_timestamp,
                             model=model,
                             params=model.get_params(),
                             r2=model_r2,
                             predictions=model_predictions,
                             mae=model_mae,
                             mape=model_mape,
                             accuracy=model_accuracy,
                             fi_all=model_fi_all,
                             fi_most_important_df=model_fi_most_important_df,
                             fi_most_important_list=model_fi_most_important_list)

        if self.verbose > 0:
            print(f"{_id_method}Target column:  {self.target_col}  ({model_results['targets'].size} values)\n"
                  f"{_id_method}Number of features used in model:  {len(model_results['feature_names'])}\n"
                  f"{_id_method}Names of features used in model:  {model_results['feature_names']}\n"
                  f"{_id_method}Model parameters:  {model_results['params']}\n"
                  f"{_id_method}Model MAE:  {model_results['mae']}\n"
                  f"{_id_method}Model MAPE:  {model_results['mape']:.3f}%\n"
                  f"{_id_method}Model R2:  {model_results['r2']}\n"
                  f"{_id_method}Model features:\n"
                  f"{model_results['fi_most_important_df']}\n"
                  f"{_id_method}{'=' * 30} END\n")

        return model_results

    def _fill(self):
        """
        Fill gaps in target

        The model used for gapfilling was built during feature
        reduction. Only rows where all features are available
        can be gapfilled.

        This means that predictions might not be available for
        some rows, namely those where at least one feature is
        missing.

        """

        _id_method = "[GAPFILLING]    "

        if self.verbose > 0:
            print(f"\n\n{_id_method}START {'=' * 30}")

        df = self.df.copy()

        # Targets w/ full timestamp, gaps in this series will be filled
        targets_ser = pd.Series(data=df[self.target_col], index=df.index)

        # Remove target from df
        df.drop(self.target_col, axis=1, inplace=True)

        # df now contains features only, keep rows with complete features
        df = df.dropna()

        # Data as arrays
        features = np.array(df)  # Features (used to predict target) need to be complete
        timestamp = df.index  # Timestamp for features and therefore predictions

        # Column names
        feature_names = list(df.columns)

        # Apply model to predict target (model already exists from previous steps)
        predictions = self.model.predict(X=features)  # Predict targets with features
        # _model_r2 = self.feat_reduction_model.score(X=_features, y=_targets_series.values)

        # Importances
        _feat_importances_all_list, \
        most_important_feat_df, \
        _most_important_feat_list = \
            model_importances(model=self.model,
                              feature_names=feature_names,
                              threshold_important_features=None)

        # Collect gapfilling results in df
        self._define_cols()  # Define column names for gapfilled_df
        gapfilled_df = self._collect(predictions=predictions,
                                     timestamp=timestamp,
                                     targets_series=targets_ser)

        # Fill still existing gaps (fallback)
        # Build model exclusively from timestamp features. Here, features are trained
        # on the already gapfilled time series.
        _still_missing_locs = gapfilled_df[self.target_gapfilled_col].isnull()
        _num_still_missing = _still_missing_locs.sum()
        if _num_still_missing > 0:
            fallback_predictions, fallback_timestamp = \
                self._gapfilling_fallback(series=gapfilled_df[self.target_gapfilled_col])
            fallback_series = pd.Series(data=fallback_predictions, index=fallback_timestamp)
            gapfilled_df[self.target_gapfilled_col].fillna(fallback_series, inplace=True)
            gapfilled_df[self.predictions_fallback_col] = fallback_series
            gapfilled_df.loc[_still_missing_locs, self.target_gapfilled_flag_col] = 2  # Adjust flag
        else:
            gapfilled_df[self.predictions_fallback_col] = None

        # Cumulative
        gapfilled_df[self.target_gapfilled_cumu_col] = \
            gapfilled_df[self.target_gapfilled_col].cumsum()

        # Store results
        gf_results = dict(
            feature_names=feature_names,
            num_features=len(feature_names),
            first_timestamp=gapfilled_df.index[0],
            last_timestamp=gapfilled_df.index[-1],
            max_potential_vals=len(gapfilled_df.index),
            target_numvals=gapfilled_df[self.target_col].count(),
            target_numgaps=gapfilled_df[self.target_col].isnull().sum(),
            target_gapfilled_numvals=gapfilled_df[self.target_gapfilled_col].count(),
            target_gapfilled_numgaps=gapfilled_df[self.target_gapfilled_col].isnull().sum(),
            target_gapfilled_flag_notfilled=int((gapfilled_df[self.target_gapfilled_flag_col] == 0).sum()),
            target_gapfilled_flag_with_hq=int((gapfilled_df[self.target_gapfilled_flag_col] == 1).sum()),
            target_gapfilled_flag_with_fallback=int((gapfilled_df[self.target_gapfilled_flag_col] == 2).sum()),
            predictions_hq_numvals=gapfilled_df[self.predictions_col].count(),
            predictions_hq_numgaps=gapfilled_df[self.predictions_col].isnull().sum(),
            predictions_fallback_numvals=gapfilled_df[self.predictions_fallback_col].count(),
            predictions_fallback_numgaps=gapfilled_df[self.predictions_fallback_col].isnull().sum(),
            model=self.model
        )

        if self.verbose > 0:
            for key, val in gf_results.items():
                print(f"{_id_method}{key}:  {val}")
            print(f"{_id_method}{'=' * 30} END\n")

        return gapfilled_df, gf_results

    def _gapfilling_fallback(self, series: pd.Series):
        """Fill data gaps using timestamp features only, fallback for still existing gaps"""
        gf_fallback_df = pd.DataFrame(series)
        gf_fallback_df = include_timestamp_as_cols(df=gf_fallback_df)

        # Build model for target predictions *from timestamp*
        # Only model-building, no predictions in this step.
        targets, features, _, _ = \
            frames.convert_to_arrays(df=gf_fallback_df,
                                     complete_rows=True,
                                     target_col=self.target_gapfilled_col)

        # Build model based in timestamp features
        model, model_r2 = \
            train_random_forest_regressor(features=features,
                                          targets=targets,
                                          n_estimators=self.n_estimators,
                                          model_params=self.model_params)

        # Use model to predict complete time series
        _, features, _, timestamp = \
            frames.convert_to_arrays(df=gf_fallback_df,
                                     complete_rows=False,
                                     target_col=self.target_gapfilled_col)
        # Predict targets w/ features
        predictions = \
            model.predict(X=features)

        return predictions, timestamp

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

    def _collect(self, predictions, timestamp, targets_series):
        # Make series w/ timestamp for predictions
        predicted_series = pd.Series(data=predictions, index=timestamp, name=self.predictions_col)

        # Reindex predictions to same timestamp as targets
        predicted_series = predicted_series.reindex(targets_series.index)

        # Collect predictions and targets in df
        gapfilled_df = \
            pd.concat([targets_series, predicted_series], axis=1)

        # Gap locations
        # Make column that contains predicted values for rows
        # where target is missing.
        _gap_locs = gapfilled_df[self.target_col].isnull()
        gapfilled_df[self.predicted_gaps_col] = \
            gapfilled_df.loc[_gap_locs, self.predictions_col]

        # Flag
        # Make flag column that indicates where predictions for
        # missing targets are available.
        # todo Note that missing predicted gaps = 0. change?
        _gapfilled_locs = gapfilled_df[self.predicted_gaps_col].isnull()  # Non-gapfilled locations
        _gapfilled_locs = ~_gapfilled_locs  # Inverse for gapfilled locations
        gapfilled_df[self.target_gapfilled_flag_col] = _gapfilled_locs
        gapfilled_df[self.target_gapfilled_flag_col] = gapfilled_df[self.target_gapfilled_flag_col].astype(int)

        # Gap-filled time series
        gapfilled_df[self.target_gapfilled_col] = \
            gapfilled_df[self.target_col].fillna(gapfilled_df[self.predictions_col])

        return gapfilled_df

    def _define_cols(self):
        self.predictions_col = ".predictions"
        self.predicted_gaps_col = ".gap_predictions"
        self.target_gapfilled_col = f"{self.target_col}_gfRF"
        self.target_gapfilled_flag_col = f"QCF_{self.target_gapfilled_col}"  # "[0=measured]"
        self.predictions_fallback_col = ".predictions_fallback"
        self.target_gapfilled_cumu_col = ".gapfilled_cumulative"


if __name__ == '__main__':
    pass
    # # For testing purposes
    # from pathlib import Path
    #
    # sourcefile = Path('../tests/test.csv')
    # source_df = pd.read_csv(sourcefile, header=[0, 1], parse_dates=True, index_col=0)
    # target_col = ('target', '-')
    #
    # # todo TESTING
    # _testcols = [('SWC_0.05', '-'),
    #              ('TS_0.05', '-'),
    #              target_col]
    # source_df = source_df[_testcols].copy()
    # # todo TESTING
    #
    # rfts = RandomForestTS(df=source_df,
    #                       target_col=target_col,
    #                       verbose=1,
    #                       random_state=42,
    #                       rfecv_step=1,
    #                       rfecv_min_features_to_select=5,
    #                       rf_rfecv_n_estimators=3,
    #                       n_estimators=3,
    #                       bootstrap=True)
    #
    # donotlag_cols = []
    # [donotlag_cols.append(x) for x in source_df.columns if '.timesince' in x[0]]
    # # [donotlag_cols.append(x) for x in source_df.columns if str(x[0]).startswith('.')]
    # target_col = ('target', '-')
    # donotlag_cols.append(target_col)
    #
    # # rfts.include_timestamp_as_features(doy_as_feature=True,
    # #                                    week_as_feature=True,
    # #                                    month_as_feature=True,
    # #                                    hour_as_feature=True)
    #
    #
    # # Model for gapfilling
    # rfts.build_final_model()
    #
    # # Gapfilling
    # rfts.gapfilling()
    # gapfilled_df, gf_results = rfts.get_gapfilled_dataset()
    #
    # # Plot
    # import matplotlib.pyplot as plt
    #
    # _scatter_df = gapfilled_df[[('target', '-'), ('.predictions', '[aux]')]].dropna()
    # _scatter_df.plot.scatter(('target', '-'), ('.predictions', '[aux]'))
    # plt.show()
    #
    # rfts.gapfilled_df.plot(subplots=True, figsize=(16, 9))
    # plt.show()
