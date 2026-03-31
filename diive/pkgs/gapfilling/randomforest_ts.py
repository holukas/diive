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
            vectorize_timestamps=vectorize_timestamps,
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
            vectorize_timestamps=True,
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
    # # Setup, user settings
    # TARGET_COL = 'NEE_CUT_REF_orig'
    # subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
    # from diive.configs.exampledata import load_exampledata_parquet_long
    # df_orig = load_exampledata_parquet_long()
    # df = df_orig.copy()
    # keep = (df.index.year >= 1997) & (df.index.year <= 2001)
    # df = df[keep].copy()
    # df = df[subsetcols].copy()

    # Load data
    from diive.core.io.files import save_parquet, load_parquet
    filepath = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\30_FLUX_PROCESSING_CHAIN\33_IRGA75_SHC_2004-2017_2019\TESTING.parquet"
    df = load_parquet(filepath=filepath)
    TARGET_COL = 'NEE_L3.1_L3.3_CUT_50_QCF0'
    subsetcols = [TARGET_COL, "TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]
    df = df[subsetcols].copy()

    # # TimeSince
    # from diive.pkgs.createvar.timesince import TimeSince
    # ts = TimeSince(df['PREC_TOT_T1_25+20_1'], upper_lim=None, lower_lim=0, include_lim=False)
    # ts.calc()
    # ts_full_results = ts.get_full_results()
    # df['TIMESINCE_PREC_TOT_T1_25+20_1'] = ts_full_results['TIMESINCE_PREC_TOT_T1_25+20_1'].copy()
    # df = df.drop('PREC_TOT_T1_25+20_1', axis=1)

    N_ESTIMATORS = 5
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
        # vectorize_timestamps=False,
        vectorize_timestamps=True,
        # add_continuous_record_number=False,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
        perm_n_repeats=10,
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

    print(rfts.feature_importances_)
    print(rfts.scores_)
    print(rfts.gapfilling_df_)

    # # Plot
    # title = (
    #     f"N_ESTIMATORS: {N_ESTIMATORS} "
    #     f"/ MAX_DEPTH: {MAX_DEPTH} "
    #     f"/ CRITERION: {CRITERION} "
    #     f"\nMIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT} "
    #     f"/ MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}  "
    # )
    # from diive.core.plotting.timeseries import TimeSeries
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # TimeSeries(series=gapfilled.multiply(0.02161926).cumsum(), ax=ax).plot(color='blue')
    # fig.suptitle(f'RF {title}', fontsize=16)
    # # ax.set_ylim(-2000, 200)
    # fig.show()

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=gapfilled).show()

    from diive.core.plotting.cumulative import CumulativeYear
    CumulativeYear(
        series=gapfilled.multiply(0.02161926),
        series_units="units",
        yearly_end_date=None,
        # yearly_end_date='08-11',
        start_year=1997,
        end_year=2022,
        show_reference=True,
        excl_years_from_reference=None,
        # excl_years_from_reference=[2022],
        # highlight_year=2022,
        highlight_year_color='#F44336').plot(digits_after_comma=0)

    from diive.core.plotting.cumulative import Cumulative
    Cumulative(df=pd.DataFrame(gapfilled).multiply(0.02161926)).plot()

    # from diive.core.plotting.dielcycle import DielCycle
    # series = gapfilled.multiply(0.02161926).copy()
    # # for yr in [2004, 2006, 2015, 2022]:
    # for yr in range(1997, 2002):
    #     series1 = series.loc[series.index.year == yr].copy()
    #     dc = DielCycle(series=series1)
    #     dc.plot(ax=None, title=str(yr), txt_ylabel_units="units",
    #             each_month=True, legend_n_col=2, ylim=[-0.4, 0.2])
    #     # d = dc.get_data()

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
