import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from diive.core.ml.common import MlRegressorGapFillingBase
from diive.core.plotting.styles.LightTheme import colorwheel_36_wider
from diive.core.times.neighbors import neighboring_years


class LongTermGapFillingBase:

    def __init__(self,
                 regressor,
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
        Gap-fill each year based on data from the respective year and the two closest neighboring years

        Example:
            Given a long-term time series comprising data between 2013-2017:
                - for 2013, the model is built from 2013, 2014 and 2015 data
                - for 2014, the model is built from 2013, 2014 and 2015 data
                - for 2015, the model is built from 2014, 2015 and 2016 data
                - for 2016, the model is built from 2015, 2016 and 2017 data
                - for 2017, the model is built from 2015, 2016 and 2017 data

        Args:
            See docstring for pkgs.gapfilling.randomforest_ts.RandomForestTS

        Attributes:
            gapfilling_df_: dataframe, gapfilling results from all years in one dataframe
            gapfilled_: series, gap-filled target series from all years in one time series
            results_yearly_: dict, detailed results for each year
            scores_: dict, scoring results for each year
            feature_importances_: dict, feature importances for each year
        """
        self.regressor = regressor
        self.input_df = input_df
        self.target_col = target_col
        self.verbose = verbose
        self.perm_n_repeats = perm_n_repeats
        self.test_size = test_size
        self.features_lag = features_lag
        self.features_lagmax = features_lagmax
        self.include_timestamp_as_features = include_timestamp_as_features
        self.add_continuous_record_number = add_continuous_record_number
        self.sanitize_timestamp = sanitize_timestamp
        self.kwargs = kwargs

        self._yearpools = None
        self._results_yearly = {}
        self._gapfilling_df = pd.DataFrame()
        self._scores = {}
        self._feature_importances_yearly = {}
        self._gapfilled = pd.Series()

        # List of features that are selected in at least one year after feature reduction
        self._features_reduced_across_years = []

        self._feature_ranks_per_year = pd.DataFrame()
        self._feature_importance_per_year = pd.DataFrame()

        self._init_input_df()


    @property
    def gapfilling_df_(self) -> DataFrame:
        """Return gapfilling results from all years in one dataframe"""
        if self._gapfilling_df.empty:
            raise Exception(f'Not available: collected gap-filled data for all years.')
        return self._gapfilling_df

    @property
    def gapfilled_(self) -> Series:
        """Return gap-filled target series from all years in one time series"""
        if not isinstance(self._gapfilled, Series):
            raise Exception(f'Not available: collected gap-filled data for all years.')
        return self._gapfilled

    @property
    def results_yearly_(self) -> dict:
        """Return detailed results for each year"""
        if not self._results_yearly:
            raise Exception(f'Not available: yearly model results.')
        return self._results_yearly

    @property
    def scores_(self) -> dict:
        """Return scoring results for each year"""
        if not self._scores:
            raise Exception(f'Not available: collected scores.')
        return self._scores

    @property
    def feature_importances_yearly_(self) -> dict:
        """Return feature importances for each year."""
        if not self._feature_importances_yearly:
            raise Exception(f'Not available: collected scores.')
        return self._feature_importances_yearly

    @property
    def yearpools(self) -> dict:
        """Return model data for each year."""
        if not self._yearpools:
            raise Exception(f'Not available: yearpools.')
        return self._yearpools

    @property
    def features_reduced_across_years(self) -> list:
        """Return features after reduction that were selected in at least one year."""
        if not self._features_reduced_across_years:
            raise Exception(f'Not available: features_reduced_across_years.')
        return self._features_reduced_across_years

    @property
    def feature_ranks_per_year(self) -> DataFrame:
        """Return feature ranks per year."""
        if self._feature_ranks_per_year.empty:
            raise Exception(f'Not available: _feature_ranks_per_year.')
        return self._feature_ranks_per_year

    @property
    def feature_importance_per_year(self) -> DataFrame:
        """Return feature importance per year."""
        if self._feature_importance_per_year.empty:
            raise Exception(f'Not available: _feature_importance_per_year.')
        return self._feature_importance_per_year

    def _init_input_df(self):
        # todo
        # Add additional variables across all years
        temp = MlRegressorGapFillingBase(
            regressor=self.regressor,
            input_df=self.input_df,
            target_col=self.target_col,
            verbose=self.verbose,
            features_lag=self.features_lag,
            include_timestamp_as_features=self.include_timestamp_as_features,
            add_continuous_record_number=self.add_continuous_record_number,
            sanitize_timestamp=self.sanitize_timestamp
        )

        # Update input data with added variables
        self.input_df = temp.model_df

    def create_yearpools(self):
        """For each year create a dataset comprising the respective year
        and the two neighboring years."""

        # Assign data for each year
        self._yearpools = neighboring_years(df=self.input_df, verbose=True)

    def initialize_yearly_models(self):
        """Initialize model for each year"""

        # Initialize a separate model for each year
        # Since additional variables were already added in the previous step,
        # the respective kwargs can be set to False.
        for year, _df in self._yearpools.items():
            print(f"Initializing model for {year} ...")
            df = _df['df'].copy()
            # Random forest
            # model = RandomForestTS(
            #     input_df=df,
            #     target_col=self.target_col,
            #     verbose=self.verbose,
            #     perm_n_repeats=self.perm_n_repeats,
            #     # features_lag=False,
            #     include_timestamp_as_features=False,
            #     add_continuous_record_number=False,
            #     sanitize_timestamp=False,
            #     **self.kwargs
            # )

            model = MlRegressorGapFillingBase(
                regressor=self.regressor,
                input_df=df,
                target_col=self.target_col,
                verbose=self.verbose,
                perm_n_repeats=self.perm_n_repeats,
                # features_lag=False,
                include_timestamp_as_features=False,
                add_continuous_record_number=False,
                sanitize_timestamp=False,
                **self.kwargs
            )

            # super().__init__(
            #     regressor=RandomForestRegressor,
            #     input_df=input_df,
            #     target_col=target_col,
            #     verbose=verbose,
            #     perm_n_repeats=perm_n_repeats,
            #     test_size=test_size,
            #     features_lag=features_lag,
            #     include_timestamp_as_features=include_timestamp_as_features,
            #     add_continuous_record_number=add_continuous_record_number,
            #     sanitize_timestamp=sanitize_timestamp,
            #     **kwargs
            # )

            self._results_yearly[year] = model

    def reduce_features_across_years(self):
        """Reduce features and detect features that were selected in at least one year."""
        features_reduced_across_years = []
        for year, year_dict in self._yearpools.items():
            print(f"---\nReducing features based on permutation importance for year {year} ...")
            rfts = self.results_yearly_[year]
            rfts.reduce_features()
            features_reduced_across_years.append(rfts.accepted_features_)
        # Flatten common_features (list of lists)
        features_reduced_across_years = [item for sublist in features_reduced_across_years for item in sublist]
        self._features_reduced_across_years = list(set(features_reduced_across_years))
        print(f"---")
        print(f"Finished reducing features based on permutation importance for all years.")
        print(f"List of features after reduction: {self.features_reduced_across_years}.")
        print(f"Each feature was selected in at least one year.")
        print(f"---")

        # Keep features that were selected
        keepcols = self.features_reduced_across_years.copy()
        keepcols.append(self.target_col)
        self.input_df = self.input_df[keepcols].copy()

        # Re-setup yearpools and models with the updated features
        self.create_yearpools()
        self.initialize_yearly_models()

    def fillgaps(self):
        """Train model for each year"""
        for year, _df in self._yearpools.items():
            print(f"Training model for {year} ...")
            rfts = self.results_yearly_[year]
            rfts.trainmodel(showplot_scores=False, showplot_importance=False)
            print(f"Gap-filling {year} ...")
            rfts.fillgaps(showplot_scores=False, showplot_importance=False)
        self._collect()

    def showplot_feature_ranks_per_year(self):

        # kudos: https://stackoverflow.com/questions/68095438/how-to-make-a-bump-chart
        # kudos: https://stackoverflow.com/questions/57923198/way-to-change-only-the-width-of-marker-in-scatterplot-but-not-height
        width = 2
        height = 1
        verts = list(zip([-width, width, width, -width], [-height, -height, height, height]))
        colors = colorwheel_36_wider()
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(ylim=(0.5, 0.5 + len(self.feature_ranks_per_year))))
        color = -1
        for ix, row in self.feature_ranks_per_year.iterrows():
            color += 1
            ax.plot(row.index, row.values, "o", marker=verts, ms=20, color=colors[color], zorder=99)
            ax.plot(row.index, row.values, "-", marker='none', ms=20, color=colors[color], zorder=1,
                    alpha=0.5)
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        firstcol_name = self.feature_ranks_per_year.iloc[:, 0].name
        lastcol_name = self.feature_ranks_per_year.iloc[:, -1].name
        ax.yaxis.set_major_locator(FixedLocator(self.feature_ranks_per_year[firstcol_name].to_list()))
        ax.yaxis.set_major_formatter(FixedFormatter(self.feature_ranks_per_year.index.to_list()))
        yax2 = ax.secondary_yaxis("right")
        yax2.yaxis.set_major_locator(FixedLocator(self.feature_ranks_per_year[lastcol_name].to_list()))
        yax2.yaxis.set_major_formatter(FixedFormatter(self.feature_ranks_per_year.index.to_list()))
        ax.grid(axis="y")
        ax.invert_yaxis()
        fig.show()

    def _collect(self):
        """Collect results"""
        for year, _df in self._yearpools.items():
            print(f"Collecting results for {year} ...")
            rfts = self.results_yearly_[year]
            keepyear = rfts.gapfilling_df_.index.year == int(year)
            self._gapfilling_df = pd.concat([self._gapfilling_df, rfts.gapfilling_df_[keepyear]], axis=0)
            self._scores[year] = rfts.scores_
            self._feature_importances_yearly[year] = rfts.feature_importances_
            gapfilled = rfts.get_gapfilled_target()
            self._gapfilled = pd.concat([self._gapfilled, gapfilled[keepyear]])

        idf = pd.DataFrame()
        for year, f in self.feature_importances_yearly_.items():
            ff = f[['PERM_IMPORTANCE']].copy()
            ff['YEAR'] = year
            ff['FEATURE'] = f.index
            ff['RANK'] = ff['PERM_IMPORTANCE'].rank(ascending=False)
            ff = ff.reset_index(drop=True, inplace=False)
            idf = pd.concat([idf, ff], axis=0, ignore_index=True)
        fi_per_year = idf.pivot(columns='YEAR', index='FEATURE', values='PERM_IMPORTANCE')
        fi_rank_per_year = idf.pivot(columns='YEAR', index='FEATURE', values='RANK')
        firstcol_name = fi_per_year.iloc[:, 0].name
        self._feature_importance_per_year = fi_per_year.sort_values(by=firstcol_name, ascending=False)
        self._feature_ranks_per_year = fi_rank_per_year.sort_values(by=firstcol_name, ascending=True)

        self._gapfilled.index.name = gapfilled.index.name
        self._gapfilled.name = gapfilled.name


class LongTermGapFillingRandomForestTS(LongTermGapFillingBase):

    def __init__(self,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: int = 0,
                 perm_n_repeats: int = 10,
                 features_lag: list = None,
                 include_timestamp_as_features: bool = False,
                 add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False,
                 **kwargs):

        super().__init__(
            regressor=RandomForestRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            features_lag=features_lag,
            include_timestamp_as_features=include_timestamp_as_features,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            perm_n_repeats=perm_n_repeats,
            **kwargs
        )


class LongTermGapFillingXGBoostTS(LongTermGapFillingBase):

    def __init__(self,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: int = 0,
                 perm_n_repeats: int = 10,
                 features_lag: list = None,
                 include_timestamp_as_features: bool = False,
                 add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False,
                 **kwargs):

        super().__init__(
            regressor=XGBRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            features_lag=features_lag,
            include_timestamp_as_features=include_timestamp_as_features,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            perm_n_repeats=perm_n_repeats,
            **kwargs
        )


def example_longterm_rfts():
    # Setup, user settings
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()

    nee_mds = df['NEE_CUT_REF_f'].copy()

    # Subset
    # keep = df.index.year <= 2016
    # df = df[keep].copy()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    lowquality = df["QCF_NEE"] > 0
    import numpy as np
    df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()

    ltrf = LongTermGapFillingRandomForestTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=0,
        features_lag=[-1, -1],
        # features_lag=None,
        include_timestamp_as_features=True,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
        perm_n_repeats=3,
        n_estimators=9,
        random_state=42,
        # min_samples_split=4,
        # min_samples_leaf=2,
        n_jobs=-1
    )

    # # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    # ltrf = LongTermGapFillingXGBoostTS(
    #     input_df=df,
    #     target_col=TARGET_COL,
    #     verbose=0,
    #     # features_lag=[-1, -1],
    #     # include_timestamp_as_features=True,
    #     # add_continuous_record_number=True,
    #     sanitize_timestamp=True,
    #     # n_estimators=3,
    #     n_estimators=99,
    #     random_state=42,
    #     # booster='gbtree',  # gbtree (default), gblinear, dart
    #     # device='cpu',
    #     validate_parameters=True,
    #     # disable_default_eval_metric=False,
    #     early_stopping_rounds=10,
    #     max_depth=6,
    #     # max_delta_step=0,
    #     # subsample=1,
    #     learning_rate=0.3,
    #     # min_split_loss=0,
    #     # min_child_weight=1,
    #     # colsample_bytree=1,
    #     # colsample_bylevel=1,
    #     # colsample_bynode=1,
    #     # reg_lambda=1,
    #     # reg_alpha=0,
    #     tree_method='auto',  # auto, hist, approx, exact
    #     # scale_pos_weight=1,
    #     # grow_policy=0,
    #     grow_policy='lossguide',  # depthwise, lossguide
    #     # max_leaves=0,
    #     # max_bin=256,
    #     # num_parallel_tree=1,
    #     n_jobs=-1
    # )

    # Assign model data
    ltrf.create_yearpools()
    # print(ltrf.yearpools)
    ltrf.initialize_yearly_models()

    # # Feature reduction
    # ltrf.reduce_features_across_years()

    # Train model and fill gaps
    ltrf.fillgaps()

    # ltrf.showplot_feature_ranks_per_year()

    gapfilled_ = ltrf.gapfilled_
    gapfilling_df_ = ltrf.gapfilling_df_
    results_yearly_ = ltrf.results_yearly_
    scores_ = ltrf.scores_
    fi = ltrf.feature_importances_yearly_
    # features_reduced_across_years = ltrf.features_reduced_across_years
    feature_ranks_per_year = ltrf.feature_ranks_per_year
    feature_importance_per_year = ltrf.feature_importance_per_year

    # ltrf.run()

    scores = []
    for year, s in ltrf.scores_.items():
        print(f"{year}: r2 = {s['r2']}  MAE = {s['mae']}")
        scores.append(s['mae'])
    from statistics import mean
    print(mean(scores))

    from diive.core.plotting.timeseries import TimeSeries
    TimeSeries(series=gapfilled_.cumsum()).plot()
    # TimeSeries(series=nee_mds.cumsum()).plot()

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=gapfilled_).show()
    # HeatmapDateTime(series=nee_mds).show()


if __name__ == '__main__':
    example_longterm_rfts()