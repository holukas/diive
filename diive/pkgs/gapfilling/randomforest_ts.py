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
                 perm_n_repeats: int = 10,
                 test_size: float = 0.25,
                 features_lag: list = None,
                 features_lag_exclude_cols: list = None,
                 include_timestamp_as_features: bool = False,
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
        super().__init__(
            regressor=RandomForestRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            perm_n_repeats=perm_n_repeats,
            test_size=test_size,
            features_lag=features_lag,
            features_lag_exclude_cols=features_lag_exclude_cols,
            include_timestamp_as_features=include_timestamp_as_features,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            **kwargs
        )


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

        self.rfts = RandomForestTS(
            input_df=self.df,
            target_col=self.target_col,
            verbose=True,
            features_lag=[-1, -1],
            include_timestamp_as_features=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
            n_estimators=99,
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=None,
            perm_n_repeats=9,
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
    gapfilled = qf.get_gapfilled_target()

    # Plot
    HeatmapDateTime(series=df[TARGET_COL]).show()
    HeatmapDateTime(series=gapfilled).show()


def example_rfts():
    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    from diive.configs.exampledata import load_exampledata_parquet_long
    df_orig = load_exampledata_parquet_long()
    # df_orig = load_exampledata_parquet()

    # # Create a large gap
    # remove = df.index.year != 2014
    # # df = df.drop(df.index[100:2200])
    # df = df[remove].copy()

    # Subset

    # keep = (df_orig.index.year >= 2016) & (df_orig.index.year <= 2018)
    # keep = df_orig.index.year >= 2022
    # df = df_orig[keep].copy()
    df = df_orig.copy()

    # Checking nighttime
    nt_locs = df['Rg_f'] < 50
    nt = df[nt_locs].groupby(df[nt_locs].index.year).agg(['mean'])
    means_nt = nt['NEE_CUT_REF_f']['mean']
    # import matplotlib.pyplot as plt
    # means_nt.plot()
    # plt.show()
    mean_nt_9704 = means_nt.loc[1997:2004].mean()
    mean_nt_0613 = means_nt.loc[2006:2013].mean()
    corr_nt = mean_nt_0613 / mean_nt_9704

    nt_locs_9704 = (nt_locs.index.year >= 1997) & (nt_locs.index.year <= 2004)
    df.loc[nt_locs_9704, TARGET_COL] = df.loc[nt_locs_9704, TARGET_COL].multiply(corr_nt)
    # df.loc[nt_locs_9704, TARGET_COL].describe()
    # df[TARGET_COL].describe()

    # # Checking daytime
    # dt = df['Rg_f'] > 50
    # dt = df[dt].groupby(df[dt].index.year).agg(['mean'])
    # means_dt = dt['NEE_CUT_REF_f']['mean']
    # import matplotlib.pyplot as plt
    # means_dt.plot()
    # plt.show()
    # mean_dt_9704 = means_dt.loc[1997:2004].mean()
    # mean_dt_0613 = means_dt.loc[2006:2013].mean()
    # corr_dt = mean_dt_0613 / mean_dt_9704

    nee_mds = df['NEE_CUT_REF_f'].copy()

    # from diive.core.plotting.timeseries import TimeSeries
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # TimeSeries(series=nee_mds.multiply(0.02161926).cumsum(), ax=ax).plot(color='orange')
    # ax.set_ylim(-2000, 200)
    # fig.suptitle(f'NEE MDS', fontsize=16)
    # fig.show()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    lowquality = df["QCF_NEE"] > 0
    df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()

    # # Testing rolling stats
    # df['test'] = df['Tair_f'].rolling(window=6, closed='left').mean()
    # df['test2'] = df['SWC_FF0_0.15_1'].rolling(window=6, closed='left').mean()

    # from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    # TimeSeries(series=df[TARGET_COL]).plot()

    N_ESTIMATORS = 9
    MAX_DEPTH = None
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 1
    CRITERION = 'squared_error'  # “squared_error”, “absolute_error”, “friedman_mse”, “poisson”

    # Random forest
    rfts = RandomForestTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=True,
        # features_lag=None,
        features_lag=[-1, -1],
        # features_lag_exclude_cols=['test', 'test2'],
        # include_timestamp_as_features=False,
        include_timestamp_as_features=True,
        # add_continuous_record_number=False,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
        perm_n_repeats=3,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        # random_state=None,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        criterion=CRITERION,
        test_size=0.2,
        n_jobs=-1
    )
    # rfts.reduce_features()
    # rfts.report_feature_reduction()

    rfts.trainmodel(showplot_scores=False, showplot_importance=False)
    rfts.report_traintest()

    rfts.fillgaps(showplot_scores=False, showplot_importance=False)
    rfts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = rfts.get_gapfilled_target()

    frame = {
        nee_mds.name: nee_mds,
        gapfilled.name: gapfilled,
    }
    checkdf = pd.DataFrame.from_dict(frame, orient='columns')
    checkdf = checkdf.groupby(checkdf.index.year).agg('sum')
    checkdf['diff'] = checkdf[gapfilled.name].subtract(checkdf[nee_mds.name])
    checkdf = checkdf.multiply(0.02161926)
    print(checkdf)
    print(checkdf.sum())

    # rfts.feature_importances
    # rfts.scores
    # rfts.gapfilling_df

    # # # https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
    # # import shap
    # # explainer = shap.TreeExplainer(rfts.model_)
    # # xtest = rfts.traintest_details_['X_test']
    # # shap_values = explainer.shap_values(xtest)
    # # shap.summary_plot(shap_values, xtest)
    # # # shap.summary_plot(shap_values[0], xtest)
    # # shap.dependence_plot("Feature 12", shap_values, xtest, interaction_index="Feature 11")

    # Plot
    title = (
        f"N_ESTIMATORS: {N_ESTIMATORS} "
        f"/ MAX_DEPTH: {MAX_DEPTH} "
        f"/ CRITERION: {CRITERION} "
        f"\nMIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT} "
        f"/ MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}  "
    )
    from diive.core.plotting.timeseries import TimeSeries
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    TimeSeries(series=gapfilled.multiply(0.02161926).cumsum(), ax=ax).plot(color='blue')
    TimeSeries(series=nee_mds.multiply(0.02161926).cumsum(), ax=ax).plot(color='orange')
    fig.suptitle(f'{title}', fontsize=16)
    # ax.set_ylim(-2000, 200)
    fig.show()

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=nee_mds).show()
    # HeatmapDateTime(series=gapfilled).show()
    # diff = gapfilled.subtract(nee_mds)
    # diff.name = "diff"
    # HeatmapDateTime(series=diff).show()

    # # mds = df_orig['NEE_CUT_REF_f'].copy()
    # # mds = mds[mds.index.year >= 2016]
    # import matplotlib.pyplot as plt
    # # # rfts.gapfilling_df_['.PREDICTIONS_FALLBACK'].cumsum().plot()
    # # # rfts.gapfilling_df_['.PREDICTIONS_FULLMODEL'].cumsum().plot()
    # # # rfts.gapfilling_df_['.PREDICTIONS'].cumsum().plot()
    # rfts.get_gapfilled_target().cumsum().plot()
    # # mds.cumsum().plot()
    # # plt.legend()
    # plt.show()

    print("Finished.")


def example_optimize():
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
    # example_quickfill()
    example_rfts()
    # example_optimize()
