import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandas import Series, DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.times.times import insert_season
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.pkgs.createvar.potentialradiation import potrad


class FlagMultipleConstantUstarThresholds:

    def __init__(self, series, ustar, thresholds, threshold_labels,
                 showplot: bool = True, verbose: bool = True, idstr: str = None):
        self.series = series
        self.ustar = ustar
        self.thresholds = thresholds
        self.threshold_labels = threshold_labels
        self.showplot = showplot
        self.verbose = verbose
        self.idstr = idstr

        self._results = pd.concat([self.series, self.ustar], axis=1).copy()

    @property
    def results(self) -> DataFrame:
        """Return high-resolution detailed data with tags as dict of DataFrames."""
        if not isinstance(self._results, DataFrame):
            raise Exception("No USTAR flags available.")
        return self._results

    def get_results(self) -> DataFrame:
        return self.results

    def calc(self):
        for ix, threshold in enumerate(self.thresholds):
            idstr = f"{self.idstr}_{self.threshold_labels[ix]}" if self.idstr else f"{self.threshold_labels[ix]}"
            ust = FlagSingleConstantUstarThreshold(
                series=self.results[self.series.name],
                ustar=self.results[self.ustar.name],
                threshold=threshold,
                idstr=idstr,
                showplot=self.showplot,
                verbose=self.verbose
            )
            ust.calc()
            flag = ust.get_flag()
            self._results[flag.name] = flag.copy()


@ConsoleOutputDecorator()
class FlagSingleConstantUstarThreshold(FlagBase):
    flagid = 'USTAR'

    def __init__(self,
                 series: Series,
                 ustar: Series,
                 threshold: float,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """xxx"""
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.ustar = ustar
        self.threshold = threshold
        self.showplot = False
        self.verbose = False
        self.showplot = showplot
        self.verbose = verbose

        # if self.showplot:
        #     self.fig, self.ax, self.ax2 = self._plot_init()

    def calc(self):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=False)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        ok = (self.ustar >= self.threshold)
        ok = ok[ok].index
        rejected = (self.ustar < self.threshold)
        rejected = rejected[rejected].index
        n_outliers = len(rejected)

        if self.verbose:
            if self.verbose:
                print(f"Total found outliers for USTAR threshold {self._idstr} {self.threshold}: {len(rejected)} values")

        return ok, rejected, n_outliers


@ConsoleOutputDecorator()
class UstarThresholdConstantScenarios:
    """
    Check impact of different constant USTAR thresholds on available data

    Constant means that the threshold is the same for all data, e.g. the same for
    all years.
    ...

    Methods:
        calc(ustarthresholds:list=None, showplot: bool = False, verbose: bool = False):
            Creates timeseries of *series* in after application of different USTAR
            thresholds given in *ustarthresholds*.

    Properties:
        scenariosdf: DataFrame of the timeseries of *series* in different USTAR
            scenarios. Records of *series* where USTAR was below the respective
            threshold are set to missing.

    """

    def __init__(self, series: Series, ustar: Series, swinpot: Series):
        self.series = series
        self.ustar = ustar
        self.swinpot = swinpot
        self.showplot = False
        self.verbose = False

        self._scenariosdf = None

        # Detect daytime and nighttime from potential radiation
        self.daytime, self.nighttime = \
            daytime_nighttime_flag_from_swinpot(swinpot=swinpot, nighttime_threshold=50)

        # Convert 0/1 flags to False/True flag
        self.daytime = self.daytime == 1
        self.nighttime = self.nighttime == 1

    @property
    def scenariosdf(self):
        """Return timeseries of *series* of each USTAR threshold, values
        below the respective threshold were removed"""
        if not isinstance(self._scenariosdf, DataFrame):
            raise Exception(f'USTAR scenarios are empty. '
                            f'Solution: run .calc() to create USTAR scenarios for {self.series.name}.')
        return self._scenariosdf

    def calc(self, ustarthresholds: list = None, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        if ustarthresholds is None:
            ustarthresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.showplot = showplot
        self.verbose = verbose
        # self.reset()

        self._scenariosdf = pd.DataFrame(self.series).copy()

        # Create timeseries for each USTAR threshold
        for u in ustarthresholds:
            suffix = f"CUT{u}"
            colname = f"{self.series.name}_{suffix}"
            series_cut = self.series.copy()
            series_cut.loc[self.ustar < u] = np.nan
            self._scenariosdf[colname] = series_cut

        # Get daytime and nighttime data separately
        _scenariosdf_daytime = self._scenariosdf.loc[self.daytime].copy()
        _scenariosdf_nighttime = self._scenariosdf.loc[self.nighttime].copy()

        # total_potential = len(self._scenariosdf)

        if self.showplot: self._plot(daytimedf=_scenariosdf_daytime, nighttimedf=_scenariosdf_nighttime)

    def _plot(self, daytimedf, nighttimedf):
        # Count available records for each USTAR threshold
        counts = self._scenariosdf.describe().loc['count']
        counts_daytime = daytimedf.describe().loc['count']
        counts_nighttime = nighttimedf.describe().loc['count']

        # Create new figure
        fig = plt.figure(facecolor='white', figsize=(12, 16))
        gs = gridspec.GridSpec(4, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.1, right=0.98, top=0.95, bottom=0.05)
        ax_dtnt = fig.add_subplot(gs[0, 0])
        ax_dt = fig.add_subplot(gs[1, 0], sharex=ax_dtnt)
        ax_nt = fig.add_subplot(gs[2, 0], sharex=ax_dtnt)
        ax_stacked = fig.add_subplot(gs[3, 0], sharex=ax_dtnt)

        # Generate bar plots
        bar_dtnt = ax_dtnt.bar(counts.index, counts, label='daytime + nighttime', width=.9, fc='#9CCC65')
        bar_dt = ax_dt.bar(counts_daytime.index, counts_daytime, label='daytime', width=.9, fc='#FFA726')
        bar_nt = ax_nt.bar(counts_nighttime.index, counts_nighttime, label='nighttime', width=.9, fc='#42A5F5')

        # Show text in bar plots
        axes_lst = [ax_dtnt, ax_dt, ax_nt]
        counts_lst = [counts, counts_daytime, counts_nighttime]
        bar_lst = [bar_dtnt, bar_dt, bar_nt]
        for ix, a in enumerate(axes_lst):
            self._bartxt(ax=axes_lst[ix], counts=counts_lst[ix], bar=bar_lst[ix])
            pf.default_legend(ax=a)
            plt.setp(a.get_xticklabels(), visible=False)

        # Stacked bar
        bar_stacked = ax_stacked.bar(counts_nighttime.index, counts_nighttime,
                                     width=.9, label='nighttime', fc='#42A5F5')
        bar_stacked = ax_stacked.bar(counts_daytime.index, counts_daytime,
                                     width=.9, bottom=counts_nighttime, label='daytime', fc='#FFA726')
        pf.default_legend(ax=ax_stacked)

        # Figure title
        title = "Available values after applying different constant USTAR thresholds"
        fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.text(0.5, 0.01, 'USTAR thresholds', ha='center', size=20)
        fig.text(0.02, 0.5, 'Available values', va='center', rotation='vertical', size=20)

        fig.show()

    def _bartxt(self, ax, counts, bar):
        counts_perc = counts.div(counts[0]).multiply(100)
        for ix, rect in enumerate(bar):
            height = rect.get_height()
            # Number of values
            ax.text(rect.get_x() + rect.get_width() / 2.0, height,
                    f'{height:.0f}', size=16, ha='center', va='bottom')
            # Percentage
            ax.text(rect.get_x() + rect.get_width() / 2.0, height / 2,
                    f'{counts_perc[ix]:.0f}%', size=20, ha='center', va='bottom')

        # _bottom = np.nan  # Needed for stacking multiple bars on top of each other
        #
        #         for flag, row in _plot_df.iterrows():
        #             _flag_counts = _plot_df.loc[flag].replace(np.nan, 0)  # Needs 0 for correct counts
        #             if flag == 0:
        #                 ax.bar(labels, _flag_counts, width=0.8, label=flag)
        #                 for bar_ix, bar in enumerate(ax.patches):
        #                     # kudos: https://www.pythoncharts.com/matplotlib/stacked-bar-charts-labels/
        #
        #                     # Show flag 0 (best) counts in plot
        #                     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
        #                             f"{round(bar.get_height())}", ha='center', color='w', weight='bold', size=6)
        #
        #                     # Show labels *inside* plot
        #                     ax.text(bar.get_x() + bar.get_width() / 10, bar.get_y() / 2,
        #                             f"{labels[bar_ix]}", ha='left', color='white', weight='bold', size=7, rotation=90)
        #
        #                 _bottom = _flag_counts
        #             else:
        #                 ax.bar(labels, _flag_counts, width=0.8, bottom=_bottom, label=flag)
        #                 _bottom = _flag_counts + _bottom

        # ok, rejected = self._flagtests(threshold=threshold)
        # self.setflag(ok=ok, rejected=rejected)
        # self.setfiltered(rejected=rejected)


class UstarDetectionMPT:
    """

    USTAR THRESHOLD DETECTION
    =========================

    How thresholds are calculated
    -----------------------------
    - Data are divided into S seasons
    - In each season, data are divided into X air temperature (TA) classes
    - Each air temperature class is divided into Y ustar (USTAR) subclasses (data are binned per subclass)
    - Thresholds are calculated in each TA class
    - Season thresholds per bootstrapping run are calculated from all found TA class thresholds, e.g. the max of thresholds
    - Overall season thresholds are calculated from all season thresholds from all bootstrapping runs, e.g. the median
    - Yearly thresholds are calculated from overall season thresholds

    """

    flux_plateau_thres_perc = 99
    ta_ustar_corr_thres = 0.4
    flux_plateau_method = 'NEE > 10+10 Higher USTAR Subclasses'

    thres_class_col = 'USTAR_MPT_THRES_IN_CLASS'  # in m s-1
    thres_season_col = 'USTAR_MPT_THRES_IN_SEASON'  # in m s-1
    thres_year_col = 'USTAR_MPT_THRES_IN_YEAR'  # in m s-1
    bts_run_col = 'BTS_RUN'  # Bootstrapping run number
    results_max_bts_col = 'MAX_BTS'  # in m s-1
    results_min_bts_col = 'MIN_BTS'  # in m s-1
    results_median_bts_col = 'MEDIAN_BTS'  # in m s-1
    results_mean_bts_col = 'MEAN_BTS'  # in m s-1
    results_p16_bts_col = 'P16_BTS'  # in m s-1
    results_p84_bts_col = 'P84_BTS'  # in m s-1
    results_bts_runs_col = 'BTS_RUNS'  # in m s-1
    results_bts_results_col = 'BTS_RESULTS'  # in m s-1

    # ['Maximum of Class Thresholds', 'Minimum of Class Thresholds',
    #  'Median of Class Thresholds', 'Mean of Class Thresholds']
    define_bts_season_thres = 'Median of Class Thresholds'
    define_overall_season_thres = 'Maximum Of Bootstrap Season Thresholds'

    def __init__(
            self,
            df: DataFrame,
            nee_col: str,
            ta_col: str,
            ustar_col: str,
            ta_n_classes: int = 6,
            ustar_n_classes: int = 20,
            season_col: str = None,
            n_bootstraps: int = 100,
            swin_pot_col: str = None,
            nighttime_threshold: float = 20,
            utc_offset: int = None,
            lat: float = None,
            lon: float = None
    ):
        self.df = df.copy()
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ta_n_classes = ta_n_classes
        self.ustar_col = ustar_col
        self.ustar_n_classes = ustar_n_classes
        self.season_col = season_col
        self.n_bootstraps = n_bootstraps
        self.swin_pot_col = swin_pot_col
        self.nighttime_threshold = nighttime_threshold
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset

        self.bts_results_df = pd.DataFrame(columns=['BOOTSTRAP_RUN', 'SEASON', 'SEASON_THRES_MEDIAN',
                                                    'OVERALL_THRES_MAX'])  # Collects detailed results from bootstrapping runs
        self.results_seasons_df = pd.DataFrame()  # Collects essential results from bts runs

        # Setup
        if not self.swin_pot_col:
            self.swin_pot_col = self._calc_swin_pot()
        self.flag_dt_col, self.flag_nt_col = self._calc_nighttime_flag()
        self.season_col = self._add_season_info()
        self.workdf, self.workdf_dt, self.workdf_nt = self._assemble_work_dataframes()

    def sample_data(self, df):
        """Draw samples from data (bootstrapping)."""
        if self.n_bootstraps == 0:
            # w/o bootstrapping, the data are simply not sampled
            sampledf = df.copy()
        else:
            num_rows = df.shape[0]
            sampledf = df.sample(n=int(num_rows), replace=True)
            sampledf.sort_index(inplace=True)
        return sampledf

    def init_results_seasons_df(self):
        """Create dataframe that contains various threshold results for each season."""

        unique_seasons = self.workdf_nt[self.season_col].unique()

        results_cols = [self.thres_season_col, self.results_max_bts_col, self.results_min_bts_col,
                        self.results_median_bts_col, self.results_mean_bts_col, self.results_p16_bts_col,
                        self.results_p84_bts_col, self.results_bts_runs_col, self.results_bts_results_col]

        results_df = pd.DataFrame(index=unique_seasons, columns=results_cols)  # Reset df
        # results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

        return results_df

    def init_results_years_df(self):
        """Create dataframe that contains various threshold results for each year."""

        unique_years = self.workdf_nt.index.year.unique()

        results_cols = [self.thres_year_col, self.results_max_bts_col, self.results_min_bts_col,
                        self.results_median_bts_col, self.results_mean_bts_col, self.results_p16_bts_col,
                        self.results_p84_bts_col, self.results_bts_runs_col, self.results_bts_results_col]

        results_df = pd.DataFrame(index=unique_years, columns=results_cols)  # Reset df
        # results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

        return results_df

    def run(self):

        # Collect essential results from all bts runs
        self.results_seasons_df = self.init_results_seasons_df()  # Reset df
        self.results_years_df = self.init_results_years_df()  # Reset df

        # If data are not bootstrapped, still use the bootstrap loop but don't sample
        overall_runs = 1 if self.n_bootstraps == 0 else self.n_bootstraps

        results_so_far = []

        for bts_run in range(0, overall_runs):
            print(f"BOOTSTRAP RUN #{bts_run}: ", end=" ")

            # SAMPLE DATA
            sampledf = self.sample_data(df=self.workdf_nt)
            sampledf = sampledf.reset_index(inplace=False)  # Keeps timestamp as column in df

            # SEPARATE BY SEASON DATA POOL
            # Loop through all available seasons
            season_counter = -1  # Count if first, second, ... season in this bts run
            season_grouped_df = sampledf.groupby(self.season_col)
            for season_key, season_df in season_grouped_df:
                # # todo---
                # # todo testing XGB and RF
                # # testdf = self.workdf['.SEASON'] == 1
                # # testdf = self.workdf_nt.loc[testdf, ['USTAR', 'NEE_L3.1_L3.2_QCF']].copy()
                # from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
                # gf = RandomForestTS(
                #     input_df=season_df[['USTAR', 'NEE_L3.1_L3.2_QCF']].copy(),
                #     target_col='NEE_L3.1_L3.2_QCF',
                #     verbose=True,
                #     features_lag=None,
                #     # features_lag=[-1, -1],
                #     # features_lag_exclude_cols=['test', 'test2'],
                #     include_timestamp_as_features=False,
                #     # include_timestamp_as_features=True,
                #     add_continuous_record_number=False,
                #     # add_continuous_record_number=True,
                #     sanitize_timestamp=False,  # todo !!!
                #     perm_n_repeats=3,
                #     n_estimators=9,
                #     random_state=42,
                #     # random_state=None,
                #     max_depth=None,
                #     min_samples_split=20,
                #     min_samples_leaf=10,
                #     # criterion=CRITERION,
                #     test_size=0.2,
                #     n_jobs=-1
                # )
                # # from diive.pkgs.gapfilling.xgboost_ts import XGBoostTS
                # # gf = XGBoostTS(
                # #     input_df=self.workdf_nt[['USTAR', 'NEE_L3.1_L3.2_QCF']].copy(),
                # #     target_col='NEE_L3.1_L3.2_QCF',
                # #     verbose=1,
                # #     # features_lag=None,
                # #     features_lag=[-1, -1],
                # #     # features_lag_exclude_cols=['Rg_f', 'TA>0', 'TA>20', 'DAYTIME', 'NIGHTTIME'],
                # #     # include_timestamp_as_features=False,
                # #     include_timestamp_as_features=True,
                # #     # add_continuous_record_number=False,
                # #     add_continuous_record_number=True,
                # #     sanitize_timestamp=True,
                # #     perm_n_repeats=3,
                # #     n_estimators=200,
                # #     random_state=42,
                # #     n_jobs=-1
                # # )
                # # xgbts.reduce_features()
                # # xgbts.report_feature_reduction()
                # gf.trainmodel(showplot_scores=False, showplot_importance=False)
                # # xgbts.report_traintest()
                # gf.fillgaps(showplot_scores=False, showplot_importance=False)
                # gf.report_gapfilling()
                # r = gf.gapfilling_df_
                # rr = self.workdf_nt[['USTAR']].copy()
                # pred = r['.PREDICTIONS_FULLMODEL']
                # rr['pred'] = pred
                # # rr.plot.scatter('USTAR', 'pred', alpha=.1)
                # # plt.xlim([0, .4])
                # # plt.ylim([0, 5])
                # # plt.show()
                #
                # from diive.core.plotting.scatter import ScatterXY
                # ScatterXY(x=rr['USTAR'], y=rr['pred'], nbins=20, ylim=[0, 20]).plot()
                # ScatterXY(x=rr['USTAR'], y=rr['pred'], nbins=50, ylim=[0, 10]).plot()
                #
                # rrr = rr.copy()
                # rrr = rrr.dropna()
                # rrr = rrr.sort_values(by=['USTAR'], ascending=False)
                # rrr = rrr.rolling(window=10, center=True).median()
                # rrr = rrr.dropna()
                # rrr.plot.scatter('USTAR', 'pred')
                # plt.show()
                # # todo---

                season_counter += 1

                # ASSIGN TA CLASSES and USTAR SUBCLASSES
                # Create df with assigned TA and USTAR classes for this season.
                season_df, ta_class_col, ustar_subclass_col = \
                    self._assign_classes(season_df=season_df,
                                         class_col=self.ta_col,
                                         subclass_col=self.ustar_col,
                                         n_classes=self.ta_n_classes,
                                         n_subclasses=self.ustar_n_classes)

                # CALCULATE SEASON THRESHOLDS
                # Calculate average values in ustar subclasses
                season_subclass_median_df = \
                    self.calculate_subclass_medians(season_df=season_df,
                                                    ta_class_col=ta_class_col,
                                                    ustar_subclass_col=ustar_subclass_col,
                                                    season_key=season_key)

                # Detect thresholds in each air temperature class
                season_subclass_median_df = \
                    self.detect_class_thresholds(season_subclass_avg_df=season_subclass_median_df,
                                                 ta_class_col=ta_class_col,
                                                 season_key=season_key)

                # Set season thresholds based on results in temperature classes
                this_ustar_thres_season = \
                    self.set_season_threshold(season_subclass_avg_df=season_subclass_median_df,
                                              season_key=season_key)

                listt = [bts_run, season_key, this_ustar_thres_season, np.nan]
                self.bts_results_df.loc[len(self.bts_results_df)] = listt

                # self.bts_results_df.loc[season_key, 'BOOTSTRAP_RUN'] = bts_run
                # self.bts_results_df.loc[season_key, 'SEASON'] = season_key
                # self.bts_results_df.loc[season_key, 'SEASON_THRES_MEDIAN'] = this_ustar_thres_season

                # COLLECT SEASON RESULTS
                # this_bts_results_df = self.prepare_results_for_collection(df=season_subclass_median_df,
                #                                                           bts_run=bts_run)
                # self.bts_results_df = self.bts_results_df.append(this_bts_results_df)
                # self.bts_results_df = pd.concat([self.bts_results_df, this_bts_results_df], axis=0)

            # self.results_seasons_df = self.collect_season_results(bts_results_df=self.bts_results_df,
            #                                                       season_key=season_key,
            #                                                       results_df=self.results_seasons_df)

            locs = self.bts_results_df['BOOTSTRAP_RUN'] == bts_run
            season_max = self.bts_results_df.loc[locs, 'SEASON_THRES_MEDIAN'].max()
            self.bts_results_df.loc[locs, 'OVERALL_THRES_MAX'] = season_max
            print(f"{season_max:.3f} m s-1 ", end=" ")

            results_so_far.append(season_max)
            print(f"overall so far: {np.median(results_so_far):.3f}")

        # print(self.bts_results_df)

        bts_thresholds = self.bts_results_df['OVERALL_THRES_MAX'].unique()
        print(np.quantile(bts_thresholds, .16))
        print(np.quantile(bts_thresholds, .50))
        print(np.quantile(bts_thresholds, .84))

        # # CALCULATE YEARLY THRESHOLDS
        # # After last season of current bts_run is reached, calculate the yearly threshold(s)
        # self.daynight_ustar_fullres_df = self.set_yearly_thresholds(df=self.daynight_ustar_fullres_df)
        #
        # yearly_thresholds_df = self.collect_yearly_thresholds(bts_run=bts_run)
        #
        # self.results_years_df = self.collect_year_results(yearly_thresholds_df=yearly_thresholds_df,
        #                                                   results_years_df=self.results_years_df)

    def set_yearly_thresholds(self, df):
        """Calculate yearly thresholds from season thresholds in full-resolution dataframe."""

        _df = df.copy()

        # Group data by year
        grouped_df = _df.groupby(_df.index.year)
        for year_key, year_df in grouped_df:

            # Check which thresholds were found for this year
            found_season_thresholds = year_df.loc[:, self.thres_season_col].unique()
            found_season_thresholds = found_season_thresholds[~np.isnan(found_season_thresholds)]  # Remove NaN

            if len(found_season_thresholds) > 0:
                # Set threshold for this year based on season results
                if self.define_year_thres == 'Maximum Of Season Thresholds':
                    threshold = found_season_thresholds.max()
                elif self.define_year_thres == 'Minimum Of Season Thresholds':
                    threshold = found_season_thresholds.min()
                elif self.define_year_thres == 'Median Of Season Thresholds':
                    threshold = np.median(found_season_thresholds)
                elif self.define_year_thres == 'Mean Of Season Thresholds':
                    threshold = found_season_thresholds.mean()
                else:
                    threshold = '-unknown-option-'
            else:
                # If no season thresholds for this year were found, the threshold for the year
                # is set to the selected percentile of the ustar data (Papale et al., 2006).
                threshold = _df[self.ustar_data_col].quantile(self.set_to_ustar_data_percentile)

                # Check if threshold is larger than the minimum allowed threshold.
                threshold = self.allowed_min_ustar_thres if threshold < self.allowed_min_ustar_thres else threshold

            _df.loc[_df.index.year == year_key, self.out_thres_year_col] = threshold

        # Add flag integers
        # Flag, season thresholds
        _df.loc[_df[self.ustar_data_col] > _df[self.out_thres_year_col],
        self.out_qcflag_year_col] = 0  # Good data
        _df.loc[_df[self.ustar_data_col] < _df[self.out_thres_year_col],
        self.out_qcflag_year_col] = 2  # Hard flag

        return _df

    def collect_year_results(self, yearly_thresholds_df, results_years_df):
        """Calculate different variants of year thresholds and the overall
        year threshold, based on bts results. Calculations are done afer
        each bts run."""

        years = yearly_thresholds_df.columns
        for year in years:
            found_yearly_thresholds = yearly_thresholds_df[year]

            filter_year = results_years_df.index == year

            # Different variants
            results_years_df.loc[filter_year, self.results_max_bts_col] = found_yearly_thresholds.max()
            results_years_df.loc[filter_year, self.results_min_bts_col] = found_yearly_thresholds.min()
            results_years_df.loc[filter_year, self.results_median_bts_col] = np.median(found_yearly_thresholds)
            results_years_df.loc[filter_year, self.results_mean_bts_col] = found_yearly_thresholds.mean()
            results_years_df.loc[filter_year, self.results_p25_bts_col] = np.quantile(found_yearly_thresholds, 0.25)
            results_years_df.loc[filter_year, self.results_p75_bts_col] = np.quantile(found_yearly_thresholds, 0.75)
            results_years_df.loc[filter_year, self.results_bts_runs_col] = self.num_bootstrap_runs
            results_years_df.loc[filter_year, self.results_bts_results_col] = len(found_yearly_thresholds)

            # Insert the overall season threshold, depending on selected option in dropdown menu
            if self.define_year_thres == 'Maximum Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_max_bts_col]
            elif self.define_year_thres == 'Minimum Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_min_bts_col]
            elif self.define_year_thres == 'Median Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_median_bts_col]
            elif self.define_year_thres == 'Mean Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_mean_bts_col]
            else:
                results_years_df.loc[filter_year, self.thres_year_col] = np.nan

        return results_years_df

    def collect_yearly_thresholds(self, bts_run):
        years = self.daynight_ustar_fullres_df.index.year.unique()
        if bts_run == 0:  # TODO hier weiter
            yearly_thresholds_df = pd.DataFrame(columns=years)  # Reset df
        grouped_df = self.daynight_ustar_fullres_df.groupby(self.daynight_ustar_fullres_df.index.year)
        for year_key, year_df in grouped_df:
            threshold = year_df[self.thres_year_col].unique()
            threshold = threshold[~np.isnan(threshold)]  # Remove NaN
            yearly_thresholds_df.loc[bts_run, year_key] = float(threshold)
        return yearly_thresholds_df

    def collect_season_results(self, bts_results_df, season_key, results_df):
        """Calculate different variants of season thresholds and the overall
        season threshold, based on bts results. Calculations are done after
        each bts run."""

        # Calculate different variants of season thresholds
        found_season_thresholds = bts_results_df.loc[season_key, self.thres_season_col].unique()
        found_season_thresholds = found_season_thresholds[~np.isnan(found_season_thresholds)]  # Remove NaN
        if len(found_season_thresholds) == 0:
            return results_df

        filter_season = results_df.index == season_key  # Current season

        # Different variants
        results_df.loc[filter_season, self.results_max_bts_col] = found_season_thresholds.max()
        results_df.loc[filter_season, self.results_min_bts_col] = found_season_thresholds.min()
        results_df.loc[filter_season, self.results_median_bts_col] = np.median(found_season_thresholds)
        results_df.loc[filter_season, self.results_mean_bts_col] = found_season_thresholds.mean()
        results_df.loc[filter_season, self.results_p16_bts_col] = np.quantile(found_season_thresholds, 0.16)
        results_df.loc[filter_season, self.results_p84_bts_col] = np.quantile(found_season_thresholds, 0.84)
        results_df.loc[filter_season, self.results_bts_runs_col] = self.n_bootstraps
        results_df.loc[filter_season, self.results_bts_results_col] = len(found_season_thresholds)

        # Insert the overall season threshold, depending on selected option in dropdown menu
        if self.define_overall_season_thres == 'Maximum Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_max_bts_col]
        elif self.define_overall_season_thres == 'Minimum Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_min_bts_col]
        elif self.define_overall_season_thres == 'Median Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_median_bts_col]
        elif self.define_overall_season_thres == 'Mean Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_mean_bts_col]
        else:
            results_df.loc[filter_season, self.thres_season_col] = np.nan

        return results_df

    def prepare_results_for_collection(self, df, bts_run):
        """Narrow df to ustar results."""
        # Subset: only ustar results columns
        ustar_resultcolumns = [self.thres_class_col, self.thres_season_col]
        results_ustar_df = df[ustar_resultcolumns].copy()
        # results_ustar_df = df.loc[self.ix_slice[:, :, :], ustar_resultcolumns]
        results_ustar_df[self.bts_run_col] = bts_run
        return results_ustar_df

    def set_season_threshold(self, season_subclass_avg_df, season_key):
        """Set threshold for current season, based on class thresholds.

        Season threshold is calculated from the class thresholds.
        """

        df = season_subclass_avg_df.copy()
        df[self.thres_season_col] = np.nan
        all_ustar_thres_classes = df.loc[season_key, self.thres_class_col].dropna().unique()

        # Check if array has contents (i.e. not emtpy)
        if all_ustar_thres_classes.size:
            if self.define_bts_season_thres == 'Median of Class Thresholds':
                ustar_thres_season = np.median(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Mean of Class Thresholds':
                ustar_thres_season = np.mean(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Maximum of Class Thresholds':
                ustar_thres_season = np.max(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Minimum of Class Thresholds':
                ustar_thres_season = np.min(all_ustar_thres_classes)
            else:
                ustar_thres_season = np.median(all_ustar_thres_classes)
        else:  # if empty
            ustar_thres_season = np.nan

        # df.loc[season_key, self.thres_season_col] = ustar_thres_season

        # df.loc[self.ix_slice[season_key, :, :],
        #        self.thres_season_col] = ustar_thres_season

        return ustar_thres_season

        # # OVERALL THRESHOLD
        # # -----------------
        # # The overall threshold is the max of the season thresholds todo per year?
        # if df[self.thres_season_col].dropna().empty:
        #     # If Series is empty, this means that no USTAR threshold was found in any
        #     # of the seasons. Following Papale et al. (2006), the threshold for the year
        #     # is then set to the 90th percentile of the USTAR data.
        #     ustar_thres_overall = season_df[self.ustar_data_col].quantile(self.set_to_ustar_data_percentile)
        #
        # else:
        #     ustar_thres_overall = self.get_ustar_overall_thres(method=self.define_overall_thres,
        #                                                        df=df,
        #                                                        ustar_thres_season_col=self.thres_season_col)
        #
        # # Finally, check if the overall USTAR threshold is below the allowed limit
        # if ustar_thres_overall > self.allowed_min_ustar_thres:
        #     pass
        # else:
        #     ustar_thres_overall = self.allowed_min_ustar_thres
        #
        # df.loc[self.ix_slice[:, :, :], self.thres_year_col] = ustar_thres_overall

        # return df, self.thres_year_col, self.thres_season_col, self.thres_class_col

    def detect_class_thresholds(self, season_subclass_avg_df, ta_class_col, season_key):
        """Calculate ustar thresholds in air temperature classes

        Detection is done for each USTAR subclass in each TA class separately
        for each season.

        Analysis is done using the pandas .groupby method. In addition, a
        MultiIndex DataFrame that contains the subclass averages within each class
        needs to be accessed. Since the index of the DataFrame is a MultiIndex,
        it can be accessed via tuples, e.g. SEASON 3 with TA class 0 and with
        USTAR subclass 4 is accessed with the tuple (3,0,4). Yes, this is a bit tricky,
        but also very useful, isn't it.

        This analysis could also have been done in an earlier step, but it
        is useful to have all results in one DataFrame.

        """

        # Create df copy to work with
        df = season_subclass_avg_df.copy()
        df[self.thres_class_col] = np.nan  # Add column for class threhold
        # Reset_index is needed to avoid duplicates in index and columns during grouping
        # df.reset_index(drop=True, inplace=True)

        # # todo testing plot
        # class_grouped = df.groupby(ta_class_col)
        # fig = plt.figure(figsize=(16, 9))
        # gs = gridspec.GridSpec(1, 1)  # rows, cols
        # # gs.update(wspace=.2, hspace=.5, left=.05, right=.95, top=.95, bottom=.05)
        # ax = fig.add_subplot(gs[0, 0])
        # for class_key, class_group_df in class_grouped:
        #     class_group_df.plot.scatter('USTAR', 'NEE_L3.1_L3.2_QCF', title=f"TA: XXX", ax=ax)
        # fig.show()

        # Loop TA classes and calculate the ustar threshold in each class
        class_grouped = df.groupby(ta_class_col)
        for class_key, class_group_df in class_grouped:

            # SUBCLASSES
            n_subclasses = len(class_group_df)
            for cur_subclass_ix in list(range(0, n_subclasses)):

                # Uses the refined method of Pastorello et al. (2020)
                cur_flux = class_group_df.iloc[cur_subclass_ix][self.nee_col]  # Current subclass flux
                cur_ustar = class_group_df.iloc[cur_subclass_ix][self.ustar_col]
                nxt_subclass = cur_subclass_ix + 1
                nxt_flux = class_group_df.iloc[nxt_subclass][self.nee_col]  # Next flux
                nxt_nxt_subclass = cur_subclass_ix + 2

                # Calculate percentage of current subclass flux in comparison to following subclasses
                if nxt_nxt_subclass < n_subclasses:
                    # In comparison to mean of next 10 subclasses
                    cur_following_mean = \
                        class_group_df.iloc[nxt_subclass:nxt_subclass + 10][self.nee_col].mean()
                    cur_flux_perc = (cur_flux / cur_following_mean) * 100
                    # Check also next subclass in comparison to its next 10 subclasses
                    nxt_following_mean = \
                        class_group_df.iloc[nxt_nxt_subclass:nxt_nxt_subclass + 10][self.nee_col].mean()
                    nxt_flux_perc = (nxt_flux / nxt_following_mean) * 100
                else:
                    cur_following_mean = cur_flux_perc = '-no-more-subclasses-'
                    nxt_following_mean = nxt_flux_perc = '-no-more-next-subclasses-'

                # self.ustar_subclass_info(cur_season=season_key, cur_class=class_key,
                #                          cur_subclass=cur_subclass_ix, cur_ustar=cur_ustar,
                #                          cur_flux=cur_flux, cur_following_mean=cur_following_mean,
                #                          cur_flux_perc=cur_flux_perc, nxt_flux=nxt_flux,
                #                          nxt_following_mean=nxt_following_mean, nxt_flux_perc=nxt_flux_perc)

                if nxt_nxt_subclass == n_subclasses:
                    # When the last subclass is reached, stop for loop
                    # print(f"    *END* Last USTAR subclass {cur_subclass_ix} reached, moving to next class.")
                    break

                # Check current flux and mean of 10 following
                if not cur_flux_perc > self.flux_plateau_thres_perc:
                    continue

                # Check next flux and mean of its 10 following
                if self.flux_plateau_method == 'NEE > 10+10 Higher USTAR Subclasses':
                    if not nxt_flux_perc > self.flux_plateau_thres_perc:
                        continue

                # Check if correlation b/w TA and RH below threshold
                abs_corr = abs(class_group_df[self.ustar_col].corr(class_group_df[self.ta_col]))
                if not abs_corr < self.ta_ustar_corr_thres:
                    continue

                # Stop for loop once a ustar threshold was found
                # df.loc[[season_key, class_key, :], self.thres_class_col] = cur_ustar
                df.loc[(int(season_key), class_key), self.thres_class_col] = cur_ustar

                # df.loc[self.ix_slice[season_key, class_key, :], self.thres_class_col] = cur_ustar
                # print(f"Season: {season_key}, TA class: {class_key}, USTAR threshold set to {cur_ustar:.3f} m s-1")
                break

        return df

    def ustar_subclass_info(self, cur_season, cur_class, cur_subclass, cur_flux, cur_ustar, cur_following_mean,
                            nxt_following_mean, nxt_flux, cur_flux_perc, nxt_flux_perc):
        if cur_following_mean == '-no-more-subclasses-':
            cur_following_mean = cur_flux_perc = nxt_following_mean = nxt_flux_perc = nxt_flux = -9999

        try:
            print(f"\n\n[SEASON] {cur_season}  "
                  f"[TA CLASS] {cur_class}  "
                  f"[USTAR SUBCLASS] {cur_subclass}  "
                  f"USTAR  {cur_ustar:.2f}  "
                  f"FLUX  {cur_flux:.2f}  "
                  f"FLUX following mean {cur_following_mean:.2f}  "
                  f"FLUX/MEAN {cur_flux_perc:.2f}%  "
                  f"(NEXT) FLUX {nxt_flux:.2f}  "
                  f"(NEXT) FLUX following mean {nxt_following_mean:.2f}  "
                  f"(NEXT) FLUX/MEAN {nxt_flux_perc:.2f}  "
                  )
        except ValueError:
            print("-ValueError-")

    def calculate_subclass_medians(self, season_df, ta_class_col, ustar_subclass_col,
                                   season_key):
        """In each TA class, calculate the TA and FLUX medians in each USTAR subclass.

        Create df that contains all the subclass medians per TA class and season.

        Args:
            season_df: data for season
            ta_class_col: column name of air temperature
            ustar_subclass_col: column name of USTAR
            season_key: season identifier

        Returns:
            DataFrame with MultiIndex (3 levels)

        """

        medians_df = pd.DataFrame()

        # CLASSES (TA)
        class_grouped = season_df.groupby(ta_class_col)
        for class_key, class_group_df in class_grouped:

            # SUBCLASSES (USTAR)
            subclass_grouped = class_group_df.groupby(ustar_subclass_col)
            for subclass_key, subclass_group_df in subclass_grouped:

                median = subclass_group_df.median()
                _temp_df = pd.DataFrame(data=median)  # Convert Series to df
                _temp_df = _temp_df.T  # Transpose: Series index will be column names
                _temp_df.loc[:, self.season_col] = season_key

                if (class_key == 0) & (subclass_key == 0):
                    medians_df = _temp_df.copy()
                else:
                    medians_df = pd.concat([medians_df, _temp_df], axis=0)

        # Insert season data pool, class and subclass as pandas MultiIndex and drop respective data cols from df
        multi_ix_cols = [self.season_col, ta_class_col, ustar_subclass_col]
        _multi_ix_df = medians_df[multi_ix_cols]  # df only used for creating MultiIndex
        _multi_ix = pd.MultiIndex.from_frame(df=_multi_ix_df)
        medians_df = medians_df.set_index(_multi_ix, inplace=False)
        medians_df = medians_df.drop(multi_ix_cols, axis=1, inplace=False)  # Remove data cols that are in MultiIndex

        return medians_df

    def _assign_classes(self, season_df, class_col, subclass_col, n_classes, n_subclasses):
        """
        Generate TA classes for this season and generate USTAR subclasses for each TA class.

        :param season_df: full resolution dataframe
        :param class_col: column name to create classes
        :param subclass_col: column name to create subclasses
        :param n_classes: number of classes
        :param n_subclasses: number of subclasses
        :return:
        """
        df = season_df.copy()

        # Insert new class and subclass columns in df
        ta_class_col = 'TA_CLASS'
        ustar_subclass_col = 'USTAR_SUBCLASS'
        df[ta_class_col] = np.nan
        df[ustar_subclass_col] = np.nan

        # TA CLASSES
        # Divide season TA data into q classes of TA.
        # Quantile-based discretization function, the fact that .qcut exists is beautiful.
        df[ta_class_col] = pd.qcut(df[class_col], q=n_classes, labels=False, duplicates='drop')  # Series

        # USTAR SUBCLASSES
        class_grouped_df = df.groupby(ta_class_col)  # Group by TA class
        for class_key, class_df in class_grouped_df:  # Loop through data of each TA class
            ustar_subclass = pd.qcut(class_df[subclass_col], q=n_subclasses, labels=False, duplicates='drop')  # Series
            df[ustar_subclass_col] = df[ustar_subclass_col].combine_first(
                ustar_subclass)  # replaces NaN w/ class number

        return df, ta_class_col, ustar_subclass_col

    def _add_season_info(self):
        """Add season info to data, used for grouping data."""
        season_col = '.SEASON'
        self.df[season_col] = insert_season(timestamp=self.df.index)
        return season_col

        # else:
        #     # Use already available season column.
        #     data_df.loc[:, self.season_type_col] = self.data_df[self.season_type_col].copy()
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_type_col].copy()

        # if self.season_data_pool == 'Season Type':
        #     # Calculate one threshold per season type, e.g. one for all summers. In this case,
        #     # all data from all e.g. summers are first pooled, and then one single threshold is
        #     # calculated based on the pooled data. The threshold is then valid for all summers across
        #     # all years.
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_grouping_col].copy()
        #
        # elif self.season_data_pool == 'Season':
        #     # Calculate threshold for each season, e.g. summer 2018, summer 2019, etc. To differentiate
        #     # between multiple summers, the year is added as additional information so the seasons
        #     # can be grouped later during calcs.
        #     data_df['year_aux'] = data_df.index.year.astype(str)
        #     data_df.loc[:, self.season_data_pool_col] = \
        #         data_df['year_aux'] + '_' + data_df.loc[:, self.season_data_pool_col].astype(str)
        #     data_df.drop('year_aux', axis=1, inplace=True)

        # return data_df

    def _assemble_work_dataframes(self):
        workdf = self.df[[self.nee_col, self.ta_col, self.ustar_col,
                          self.swin_pot_col, self.flag_dt_col, self.flag_nt_col,
                          self.season_col]].copy()
        workdf = workdf.dropna()
        is_daytime = workdf[self.flag_dt_col] == 1
        is_nighttime = workdf[self.flag_nt_col] == 1
        workdf_dt = workdf[is_daytime].copy()
        workdf_nt = workdf[is_nighttime].copy()
        return workdf, workdf_dt, workdf_nt

    def _calc_nighttime_flag(self):
        flag_daytime, flag_nighttime = daytime_nighttime_flag_from_swinpot(
            swinpot=self.df[self.swin_pot_col],
            nighttime_threshold=self.nighttime_threshold
        )
        self.df[flag_daytime.name] = flag_daytime
        self.df[flag_nighttime.name] = flag_nighttime
        return flag_daytime.name, flag_nighttime.name

    def _calc_swin_pot(self):
        """Calculate potential radiation or get directly from data"""
        if self.lat and self.lon:
            swin_pot = potrad(timestamp_index=self.df.index,
                              lat=self.lat,
                              lon=self.lon,
                              utc_offset=self.utc_offset)
            swin_pot_col = swin_pot.name
            self.df[swin_pot_col] = swin_pot
            return swin_pot_col
        else:
            raise Exception("Latitude and longitude are required "
                            "if potential radiation is not in data.")


def example():
    from diive.core.io.files import load_parquet
    filepath = r"L:\Sync\luhk_work\TMP\FluxProcessingChain_L3.2.parquet"
    df = load_parquet(filepath=filepath)
    locs = (df.index.year >= 2022) & (df.index.year <= 2022)
    df = df.loc[locs].copy()
    # [print(c) for c in df.columns if "TA" in c]

    NEE_COL = "NEE_L3.1_L3.2_QCF"
    TA_COL = "TA_T1_2_1"
    USTAR_COL = "USTAR"
    SW_IN_POT_COL = None

    ust = UstarDetectionMPT(
        df=df,
        nee_col=NEE_COL,
        ta_col=TA_COL,
        ustar_col=USTAR_COL,
        ta_n_classes=6,
        ustar_n_classes=20,
        n_bootstraps=99,
        swin_pot_col=SW_IN_POT_COL,
        nighttime_threshold=20,
        utc_offset=1,
        lat=47.210227,
        lon=8.410645
    )

    ust.run()


def example_scenarios():
    # Load data
    from diive.core.io.files import load_pickle
    df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data.pickle")
    ust = UstarThresholdConstantScenarios(series=df['FC'],
                                          swinpot=df['SW_IN_POT'],
                                          ustar=df['USTAR'])
    ust.calc(ustarthresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], showplot=True, verbose=True)


def example_flag_constant_ustar_threshold():
    from diive.core.io.files import load_parquet
    filepath = r"L:\Sync\luhk_work\TMP\FluxProcessingChain_L3.2.parquet"
    df = load_parquet(filepath=filepath)
    locs = (df.index.year >= 2022) & (df.index.year <= 2022)
    df = df.loc[locs].copy()
    # [print(c) for c in df.columns if "TA" in c]

    NEE_COL = "NEE_L3.1"
    USTAR_COL = "USTAR"
    SERIES = df[NEE_COL].copy()
    USTAR = df[USTAR_COL].copy()
    # [0.0532449, 0.0709217, 0.0949867],

    ust = FlagMultipleConstantUstarThresholds(
        series=SERIES,
        ustar=USTAR,
        thresholds=[0.0532449, 0.0709217, 0.0949867],
        threshold_labels=['CUT_16', 'CUT_50', 'CUT_84'],
        showplot=True,
        verbose=True
    )
    ust.calc()

    # ust = FlagSingleConstantUstarThreshold(
    #     series=SERIES,
    #     ustar=USTAR,
    #     threshold=0.0532449,  # 0.0532449, 0.0709217, 0.0949867
    #     idstr="CUT_50",
    #     showplot=True,
    #     verbose=True
    # )
    ust.calc()
    filteredseries = ust.filteredseries
    flag = ust.get_flag()

    # df[flag.name] = flag
    #
    # # Calculate overall quality flag QCF
    # from diive.pkgs.qaqc.qcf import FlagQCF
    # qcf = FlagQCF(series=df[NEE_COL],
    #               df=df,
    #               idstr="L3.3",
    #               swinpot=df['SW_IN_POT'],
    #               nighttime_threshold=50)
    # qcf.calculate(daytime_accept_qcf_below=1,
    #               nighttimetime_accept_qcf_below=1)
    # qcf.get()
    # qcf.report_qcf_evolution()

    print("END")


if __name__ == '__main__':
    example_flag_constant_ustar_threshold()
    # example()
    # example_scenarios()
