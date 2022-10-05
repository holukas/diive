"""
=========================================
RANDOM FOREST GAP-FILLING FOR TIME SERIES
randomforest_ts
=========================================

This module is part of the diive library:
https://gitlab.ethz.ch/diive/diive

[x] lagged variants by default v0.36.0
[x] longterm filling from 3-year pools v0.36.0
[x] share already existing models v0.36.0
[x] gap-filling for all data in one model v0.36.0
[x] timeseriessplit v0.36.0
[x] later feature reduction (rfecv) v0.36.0

"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

import diive.core.dfun.frames as frames
from diive.core.funcs.funcs import find_nearest_val
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
            lagged_variants: int or None = 1,
            include_timestamp_as_features: bool = True,
            use_neighbor_years: bool = True,
            feature_reduction: bool = False,
            verbose: int = 0,
    ):
        """
        Gap-fill timeseries with random forest

        Args:
            df:
                Contains timeseries of target and features.

            target_col:
                Column name of variable that will be gap-filled.

            lagged_variants:
                Include lagged variants of the features. If and int is
                given, then variants lagged by int records will be included
                as additional features.
                Example:
                    Assuming one of the features is `TA` and `lagged_variants=1`,
                    then the new feature `TA-1` is created and added to the dataset
                    as additional feature. In this case, `TA-1` contains values from
                    the previous `TA` record (i.e., `TA` was shifted by one record).

            include_timestamp_as_features:
                If *True*, timestamp info is added as columns to the dataset. By
                default, this adds the following columns: YEAR, DOY, WEEK, MONTH
                and HOUR.

            use_neighbor_years:
                If *True*, the random forest model that is used to gap-fill data
                for a specific year is built from data comprising also data from
                the neighboring years.
                Example:
                    When gap-filling 2017, the random forest model is built from
                    2016, 2017 and 2018 data.

            feature_reduction:
                If *True*, recursive feature elimination with cross-validation
                is performed before the random forest model is built. In essence,
                this tries to reduce the number of features that are included in
                the random forest model.
                The random forest regressor is used as estimator.
                TimeSeriesSplit is used as the cross-validation splitting strategy.
                See: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
                See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

            verbose:
                If *1*, more output is printed.
        """
        self.df = df
        self.target_col = target_col
        self.verbose = verbose
        self.use_neighbor_years = use_neighbor_years
        self.feature_reduction = feature_reduction

        if lagged_variants:
            self.df = frames.steplagged_variants(df=self.df.copy(), stepsize=1, stepmax=lagged_variants,
                                                 exclude_cols=[self.target_col])

        if include_timestamp_as_features:
            self.df = include_timestamp_as_cols(df=self.df)

        if len(self.df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.df.columns}")

        self.yrpools_assignments = self._assign_yearpools()

        self.yrpool_model_results = dict()  # Contains yearpool models
        self.yrpool_gf_results = dict()  # Contains yearpool gapfilling results (info)
        self.gapfilled_yrs_df = self.df.copy()  # Will contain gapfilled data for all years

    def _assign_yearpools(self) -> dict:
        """
        Assign years to model that is used to gap-fill a specific year

        Use data from neighboring years when building model for
        a specific year

        Example:
            For gap-filling 2017, build random forest model from 2016, 2017 and 2018 data.

        Returns:
            Dict with years as keys and the list of assigned years as values.

        """
        print("Assigning years for gap-filling ...")
        uniq_years = list(self.df.index.year.unique())
        yearpools_dict = {}

        # For each year's model, use data from the neighboring years
        if self.use_neighbor_years:
            # For each year, build model from the 2 neighboring years
            # yearpool = 3
            yearpools_dict = {}

            for ix, year in enumerate(uniq_years):
                years_in_pool = []
                _uniq_years = uniq_years.copy()
                _uniq_years.remove(year)
                years_in_pool.append(year)

                if _uniq_years:
                    nearest_1 = find_nearest_val(array=_uniq_years, value=year)
                    _uniq_years.remove(nearest_1)
                    years_in_pool.append(nearest_1)

                    if _uniq_years:
                        nearest_2 = find_nearest_val(array=_uniq_years, value=year)
                        years_in_pool.append(nearest_2)

                years_in_pool = sorted(years_in_pool)

                yearpools_dict[str(year)] = years_in_pool
                print(f"Gap-filling for year {year} is based on data from years "
                      f"{years_in_pool}.")

        # Use one model for all years
        else:
            for ix, year in enumerate(uniq_years):
                years_in_pool = (uniq_years).copy()
                yearpools_dict[str(year)] = years_in_pool
                print(f"Gap-filling for year {year} is based on data from years "
                      f"{years_in_pool}.")

        return yearpools_dict

    def get_gapfilled_dataset(self) -> tuple[DataFrame, dict]:
        """

        Returns:
            Gap-filled data and info

        """
        return self.gapfilled_yrs_df, self.yrpool_gf_results

    def build_models(self, **rf_model_params):
        for yr, poolyrs in self.yrpools_assignments.items():

            model_available = False

            # Check if results and therefore a model for this timespan already exists
            for key, val in self.yrpool_model_results.items():
                if val['years_in_model'] == poolyrs:
                    self.yrpool_model_results[yr] = val
                    model_available = True
                    break

            # Build model for this year if none is already available
            if not model_available:
                # Years in model
                first_yr = min(poolyrs)
                last_yr = max(poolyrs)
                yrpool_df = self.df.loc[(self.df.index.year >= first_yr) &
                                        (self.df.index.year <= last_yr), :].copy()

                # Feature reduction
                if self.feature_reduction:
                    rfecv_results = self._rfecv(df=yrpool_df, **rf_model_params)
                    reduced_yrpool_df = yrpool_df[rfecv_results['most_important_features_after']].copy()
                    reduced_yrpool_df[self.target_col] = yrpool_df[self.target_col].copy()  # Needs target
                    self.yrpool_model_results[yr] = self._build_model(df=reduced_yrpool_df, **rf_model_params)
                else:
                    self.yrpool_model_results[yr] = self._build_model(df=yrpool_df, **rf_model_params)

    def _rfecv(self, df: DataFrame, **rf_model_params) -> dict:
        """
        Run Recursive Feature Elimination With Cross-Validation

        More info:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

        """

        _id_method = "[FEATURE REDUCTION]    "

        if self.verbose > 0:
            print(f"\n\n{_id_method}START {'=' * 30}")

        _df = df.copy()

        # Prepare data series, data can be used directly as pandas Series
        _df = _df.dropna()
        rfecv_targets = _df[self.target_col].copy()
        _df.drop(self.target_col, axis=1, inplace=True)
        rfecv_features = _df.copy()

        # Find features
        from sklearn.ensemble import RandomForestRegressor  # Import the model we are using
        estimator = RandomForestRegressor(**rf_model_params)
        splitter = TimeSeriesSplit(n_splits=5)  # Cross-validator

        rfecv = RFECV(estimator=estimator,
                      step=1,
                      min_features_to_select=3,
                      cv=splitter,
                      scoring='explained_variance',
                      verbose=self.verbose,
                      n_jobs=-1)

        # rfecv = RFECV(estimator=PermutationImportance(estimator=estimator, scoring='explained_variance',
        #                                               n_iter=10, random_state=42, cv=splitter),
        #               step=self.rfecv_step,
        #               min_features_to_select=self.rfecv_min_features_to_select,
        #               cv=splitter,
        #               scoring='explained_variance',
        #               verbose=self.verbose,
        #               n_jobs=-1)

        rfecv.fit(rfecv_features, rfecv_targets)
        # rfecv.ranking_

        # Feature importances
        rfecv_features.drop(rfecv_features.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
        rfecv_importances_df = pd.DataFrame()
        rfecv_importances_df['feature'] = list(rfecv_features.columns)
        rfecv_importances_df['importance'] = rfecv.estimator_.feature_importances_
        rfecv_importances_df = rfecv_importances_df.sort_values(by='importance', ascending=False)

        # Keep features where importance >= 0.01
        rfecv_importances_df = rfecv_importances_df.loc[rfecv_importances_df['importance'] >= 0.01, :]

        most_important_features_list = rfecv_importances_df['feature'].to_list()

        # Store results
        rfecv_results = dict(num_features_before=len(rfecv_features.columns),
                             num_features_after=len(most_important_features_list),
                             feature_importances_after=rfecv_importances_df,
                             most_important_features_after=most_important_features_list,
                             params=rfecv.get_params())

        if self.verbose > 0:
            print(f"{_id_method}Parameters:  {rfecv_results['params']}\n"
                  f"{_id_method}Number of features *before* reduction:  {len(_df.columns)}\n"
                  f"{_id_method}Number of features *after* reduction:  {rfecv_results['num_features_after']}\n"
                  f"{_id_method}Most important features:  {rfecv_results['most_important_features_after']}\n"
                  f"{rfecv_results['feature_importances_after']}\n"
                  f"{_id_method}{'=' * 30} END\n")

        return rfecv_results

    def gapfill_yrs(self):
        """Gap-fill years with their respective model"""
        for yr, model_results in self.yrpool_model_results.items():
            # Data for this year
            yr_df = self.df.loc[self.df.index.year == int(yr), :].copy()

            # Only use features that were used in building model,
            # this step also removes the target column
            reduced_yr_df = yr_df[model_results['feature_names']].copy()

            # Add target column back to data
            reduced_yr_df[self.target_col] = yr_df[self.target_col].copy()  # Needs target

            # Perform gap-filling
            gapfilled_yr_df, self.yrpool_gf_results[yr] = self._gapfill_yr(df=reduced_yr_df,
                                                                           model_results=model_results)

            # Add this year to collection of all years
            self.gapfilled_yrs_df = self.gapfilled_yrs_df.combine_first(gapfilled_yr_df)

    def _build_model(self, df: DataFrame, **rf_model_params) -> dict:
        """Build final model for gapfilling

        Parameters (default) for RandomForestRegressor:
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
            warm_start: bool = False,
            ccp_alpha: float = 0.0,  # Non-negative
            max_samples: int or float = None

        """

        _id_method = "[BUILDING MODEL]    "

        yrs_in_model = list(df.index.year.unique())

        if self.verbose > 0:
            print(f"\n{_id_method}Based on data from years {yrs_in_model} {'=' * 30}")

        # Data as arrays
        model_targets, model_features, model_feature_names, model_timestamp = \
            frames.convert_to_arrays(df=df,
                                     target_col=self.target_col,
                                     complete_rows=True)

        # Model training
        model, model_r2 = \
            train_random_forest_regressor(targets=model_targets,
                                          features=model_features,
                                          **rf_model_params)  # All features, all targets

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
                             years_in_model=yrs_in_model,
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
                  f"{_id_method}Model data first timestamp:  {model_results['timestamp'][0]}\n"
                  f"{_id_method}Model data last timestamp:  {model_results['timestamp'][-1]}\n"
                  f"{_id_method}Model parameters:  {model_results['params']}\n"
                  f"{_id_method}Model MAE:  {model_results['mae']}\n"
                  f"{_id_method}Model MAPE:  {model_results['mape']:.3f}%\n"
                  f"{_id_method}Model R2:  {model_results['r2']}\n"
                  f"{_id_method}Model features:\n"
                  f"{model_results['fi_most_important_df']}\n"
                  f"{_id_method}{'=' * 30} END\n")

        return model_results

    def _gapfill_yr(self, df: DataFrame, model_results: dict):
        """
        Fill gaps in target year

        The model used for gapfilling was built during feature
        reduction. Only rows where all features are available
        can be gapfilled.

        This means that predictions might not be available for
        some rows, namely those where at least one feature is
        missing.

        """

        uniq_years = list(df.index.year.unique())
        current_yr = uniq_years[0] if len(uniq_years) == 1 else None
        if not current_yr:
            raise Exception("(!) Data that are gap-filled comprise more than one year.")

        _id_method = f"[GAPFILLING {self.target_col} {current_yr}]    "
        model = model_results['model']

        if self.verbose > 0:
            print(f"\n\n{_id_method}START {'=' * 30}")

        # Targets w/ full timestamp, gaps in this series will be filled
        targets_ser = pd.Series(data=df[self.target_col], index=df.index)

        # Remove target from data
        df.drop(self.target_col, axis=1, inplace=True)

        # df now contains features only, keep rows with complete features
        df = df.dropna()

        # Data as arrays
        features = np.array(df)  # Features (used to predict target) need to be complete
        timestamp = df.index  # Timestamp for features and therefore predictions

        # Column names
        feature_names = list(df.columns)

        # Apply model to predict target (model already exists from previous steps)
        predictions = model.predict(X=features)  # Predict targets with features
        # _model_r2 = self.feat_reduction_model.score(X=_features, y=_targets_series.values)

        # Importances
        _feat_importances_all_list, \
        most_important_feat_df, \
        _most_important_feat_list = \
            model_importances(model=model,
                              feature_names=feature_names,
                              threshold_important_features=None)

        # Collect gapfilling results in df
        self._define_cols()  # Define column names for gapfilled_df
        gapfilled_yr_df = self._collect(predictions=predictions,
                                        timestamp=timestamp,
                                        targets_series=targets_ser)

        # Fill still existing gaps (fallback)
        # Build model exclusively from timestamp features. Here, features are trained
        # on the already gapfilled time series.
        _still_missing_locs = gapfilled_yr_df[self.target_gapfilled_col].isnull()
        _num_still_missing = _still_missing_locs.sum()
        if _num_still_missing > 0:
            fallback_predictions, fallback_timestamp = \
                self._gapfilling_fallback(series=gapfilled_yr_df[self.target_gapfilled_col],
                                          model_params=model_results['params'])
            fallback_series = pd.Series(data=fallback_predictions, index=fallback_timestamp)
            gapfilled_yr_df[self.target_gapfilled_col].fillna(fallback_series, inplace=True)
            gapfilled_yr_df[self.predictions_fallback_col] = fallback_series
            gapfilled_yr_df.loc[_still_missing_locs, self.target_gapfilled_flag_col] = 2  # Adjust flag
        else:
            gapfilled_yr_df[self.predictions_fallback_col] = None

        # Cumulative
        gapfilled_yr_df[self.target_gapfilled_cumu_col] = \
            gapfilled_yr_df[self.target_gapfilled_col].cumsum()

        # Store results
        gapfilled_yr_info = dict(
            feature_names=feature_names,
            num_features=len(feature_names),
            first_timestamp=gapfilled_yr_df.index[0],
            last_timestamp=gapfilled_yr_df.index[-1],
            max_potential_vals=len(gapfilled_yr_df.index),
            target_numvals=gapfilled_yr_df[self.target_col].count(),
            target_numgaps=gapfilled_yr_df[self.target_col].isnull().sum(),
            target_gapfilled_numvals=gapfilled_yr_df[self.target_gapfilled_col].count(),
            target_gapfilled_numgaps=gapfilled_yr_df[self.target_gapfilled_col].isnull().sum(),
            target_gapfilled_flag_notfilled=int((gapfilled_yr_df[self.target_gapfilled_flag_col] == 0).sum()),
            target_gapfilled_flag_with_hq=int((gapfilled_yr_df[self.target_gapfilled_flag_col] == 1).sum()),
            target_gapfilled_flag_with_fallback=int((gapfilled_yr_df[self.target_gapfilled_flag_col] == 2).sum()),
            predictions_hq_numvals=gapfilled_yr_df[self.predictions_col].count(),
            predictions_hq_numgaps=gapfilled_yr_df[self.predictions_col].isnull().sum(),
            predictions_fallback_numvals=gapfilled_yr_df[self.predictions_fallback_col].count(),
            predictions_fallback_numgaps=gapfilled_yr_df[self.predictions_fallback_col].isnull().sum(),
            years_in_model=model_results['years_in_model']
        )

        if self.verbose > 0:
            for key, val in gapfilled_yr_info.items():
                print(f"{_id_method}{key}:  {val}")
            print(f"{_id_method}{'=' * 30} END\n")

        return gapfilled_yr_df, gapfilled_yr_info

    def _gapfilling_fallback(self, series: pd.Series, model_params: dict):
        """Fill data gaps using timestamp features only, fallback for still existing gaps"""
        gf_fallback_df = pd.DataFrame(series)
        gf_fallback_df = include_timestamp_as_cols(df=gf_fallback_df)

        # Build model for target predictions *from timestamp*
        # Only model-building, no predictions in this step.
        targets, features, _, _ = \
            frames.convert_to_arrays(df=gf_fallback_df,
                                     complete_rows=True,
                                     target_col=self.target_gapfilled_col)

        # Build model based on timestamp features
        model, model_r2 = \
            train_random_forest_regressor(features=features,
                                          targets=targets,
                                          **model_params)

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


def example():
    import matplotlib.pyplot as plt
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime

    # # Read data file
    # from diive.core.io.filereader import ReadFileType
    # loaddatafile = ReadFileType(
    #     filetype='REDDYPROC_30MIN',
    #     filepath=r'F:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testdata_CH-DAV_FP2021.2_2016-2020_ID20220324003457_30MIN_SUBSET.csv',
    #     data_nrows=None)
    # data_df, metadata_df = loaddatafile._readfile()
    #
    # filepath = save_as_pickle(outpath=r'F:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata',
    #                filename='testdata_CH-DAV_FP2021.2_2016-2020_ID20220324003457_30MIN_SUBSET.csv',
    #                data=data_df)

    # # Test data
    # from diive.core.io.files import load_pickle
    # df = load_pickle(
    #     filepath=r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testdata_CH-DAV_FP2021.2_2016-2020_ID20220324003457_30MIN_SUBSET.csv.pickle')
    # df.drop('NEE_CUT_f', axis=1, inplace=True)

    # Download from database
    from dbc_influxdb import dbcInflux
    DIRCONF = r'L:\Dropbox\luhk_work\20 - CODING\22 - POET\configs'  # Folder with configurations
    SITE = 'ch-dav'  # Site name
    BUCKET_PROCESSING = f"{SITE}_processing"
    TARGET_COL = 'TA_NABEL_T1_35_1'
    MEASUREMENTS = ['TA', 'SW', 'RH']  # Measurement name, e.g., 'TA' contains all air temperature variables
    FIELDS = ['TA_NABEL_T1_35_1', 'SW_IN_NABEL_T1_35_1', 'RH_NABEL_T1_35_1']  # Variable name; InfluxDB stores variable names as '_field'
    DATA_VERSION = 'meteoscreening'
    START = '2021-10-01 00:01:00'  # Download data starting with this date
    STOP = '2021-12-01 00:01:00'  # Download data before this date (the stop date itself is not included)
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)

    dbc = dbcInflux(dirconf=DIRCONF)

    data_simple, data_detailed, assigned_measurements = dbc.download(bucket=BUCKET_PROCESSING,
                                                                     measurements=MEASUREMENTS,
                                                                     fields=FIELDS,
                                                                     start=START,
                                                                     stop=STOP,
                                                                     timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
                                                                     data_version=DATA_VERSION)

    # df = df.loc[df.index.year == 2016, :].copy()
    df = data_simple.copy()

    # Random forest
    rfts = RandomForestTS(df=df,
                          target_col=TARGET_COL,
                          include_timestamp_as_features=True,
                          lagged_variants=3,
                          use_neighbor_years=True,
                          feature_reduction=False,
                          verbose=1)
    rfts.build_models(n_estimators=200,
                      random_state=42,
                      min_samples_split=2,
                      min_samples_leaf=1,
                      n_jobs=-1)
    rfts.gapfill_yrs()
    gapfilled_yrs_df, yrpool_gf_results = rfts.get_gapfilled_dataset()

    # Gap-filled data series
    target_col_gf = f'{TARGET_COL}_gfRF'
    gapfilled_series = gapfilled_yrs_df[target_col_gf].copy()

    # Plot
    HeatmapDateTime(series=rfts.gapfilled_yrs_df[TARGET_COL]).show()
    HeatmapDateTime(series=rfts.gapfilled_yrs_df[target_col_gf]).show()
    # gapfilled_yrs_df['.gapfilled_cumulative'].plot()
    # plt.show()

    # Merge gapfilled series with tags
    data_detailed_gf_var = data_detailed[TARGET_COL].copy()  # Original data with tags
    data_detailed_gf_var[TARGET_COL] = gapfilled_series  # Replace original series w/ gapfilled series (same name)
    data_detailed_gf_var['data_version'] = 'gapfilled'  # Change data version to 'gapfilled'

    # Upload gapfilled var to database
    dbc.upload_singlevar(to_bucket=BUCKET_PROCESSING,
                         to_measurement=assigned_measurements[TARGET_COL],
                         var_df=data_detailed_gf_var,
                         timezone_of_timestamp='UTC+01:00')

    print("Finished.")


if __name__ == '__main__':
    example()
