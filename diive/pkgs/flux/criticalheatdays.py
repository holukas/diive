"""
FLUX: CRITICAL HEAT DAYS
========================
"""

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame

import diive.pkgs.analyses.optimumrange
from diive.common.dfun.fits import BinFitterCP
import diive.common.dfun.frames as frames
from diive.common.plotting import plotfuncs
from diive.common.plotting.fitplot import fitplot
from diive.common.plotting.rectangle import rectangle
from diive.common.plotting.styles import LightTheme as theme
from diive.pkgs.analyses.optimumrange import FindOptimumRange

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


class CriticalHeatDays:
    date_col = '.DATE'
    grp_daynight_col = '.GRP_DAYNIGHT'

    def __init__(
            self,
            df: DataFrame,
            x_col: str,
            nee_col: str,
            gpp_col: str,
            reco_col: str,
            daynight_split: str = 'timestamp',
            daynight_split_day_start_hour: int = 7,
            daynight_split_day_end_hour: int = 18,
            daytime_threshold: int = 20,
            set_daytime_if: str = 'Larger Than Threshold',
            usebins: int = 10,
            bootstrap_runs: int = 10,
            bootstrapping_random_state: int = None
    ):
        """Detect critical heat days from partitioned ecosystem fluxes

        XXX

        Args:
            df:
            x_col: Column name of x variable
            nee_col: Column name of NEE (net ecosystem exchange, ecosytem flux)
            gpp_col: Column name of GPP (gross primary production, ecosytem flux)
            reco_col: Column name of RECO (ecosystem respiration, ecosytem flux)
            daynight_split: Column name of variable used for detection of daytime and nighttime
            daytime_threshold: Threshold for detection of daytime and nighttime from *daytime_col*
            set_daytime_if: 'Larger Than Threshold' or 'Smaller Than Threshold'
            usebins: XXX (default 10)
            bootstrap_runs: XXX (default 10)
        """

        if daynight_split != 'timestamp':
            self.df = df[[x_col, nee_col, gpp_col, reco_col, daynight_split]].copy()
        else:
            self.df = df[[x_col, nee_col, gpp_col, reco_col]].copy()
        self.x_col = x_col
        self.nee_col = nee_col
        self.gpp_col = gpp_col
        self.reco_col = reco_col
        self.daynight_split = daynight_split
        self.daynight_split_day_start_hour = daynight_split_day_start_hour
        self.daynight_split_day_end_hour = daynight_split_day_end_hour
        self.daytime_threshold = daytime_threshold
        self.set_daytime_if = set_daytime_if
        self.usebins = usebins
        self.bootstrap_runs = bootstrap_runs
        self.bootstrapping_random_state = bootstrapping_random_state

        # Resample dataset
        aggs = ['mean', 'median', 'count', 'max', 'sum']
        self.df_aggs = self._resample_dataset(df=self.df, aggs=aggs, day_start_hour=7)

        # Prepare separate daytime and nighttime datasets and aggregate
        self.df_daytime, \
        self.df_daytime_aggs, \
        self.df_nighttime, \
        self.df_nighttime_aggs, = \
            self._prepare_daynight_datasets(aggs=aggs)

        # import matplotlib.pyplot as plt
        # self.df_daytime_aggs[self.nee_col]['sum'].plot()
        # plt.scatter(self.df_daytime_aggs[self.x_col]['max'],
        #             self.df_daytime_aggs[self.nee_col]['sum'])
        # plt.show()

        self.predict_min_x = self.df[x_col].min()
        self.predict_max_x = self.df[x_col].max()

        self._results_threshold_detection = {}  # Collects results for each bootstrap run
        self._results_daytime_analysis = {}
        self._results_optimum_range = {}  # Results for optimum range

    @property
    def results_threshold_detection(self) -> dict:
        """Return bootstrap results for daily fluxes, includes threshold detection"""
        if not self._results_threshold_detection:
            raise Exception('Results for threshold detection are empty')
        return self._results_threshold_detection

    @property
    def results_daytime_analysis(self) -> dict:
        """Return bootstrap results for daytime fluxes"""
        if not self._results_daytime_analysis:
            raise Exception('Results for flux analysis are empty')
        return self._results_daytime_analysis

    @property
    def results_optimum_range(self) -> dict:
        """Return results for optimum range"""
        if not self._results_optimum_range:
            raise Exception('Results for optimum range are empty')
        return self._results_optimum_range

    def analyze_daytime(self):
        """Analyze daytime fluxes"""

        # Fit to bootstrapped data, daytime data only, daily time resolution
        # Stored as bootstrap runs > 0 (bts>0)
        bts_results = self._bootstrap_fits(df_aggs=self.df_daytime_aggs,
                                           x_col=self.x_col,
                                           x_agg='max',
                                           y_cols=[self.nee_col, self.gpp_col, self.reco_col],
                                           y_agg='sum',
                                           ratio_cols=[self.gpp_col, self.reco_col],
                                           fit_to_bins=self.usebins,
                                           detect_zerocrossing=False)

        # Collect GPP:RECO ratios from the different bootstrap runs
        bts_ratios_df = pd.DataFrame()
        for bts in range(1, self.bootstrap_runs + 1):
            ratio_s = bts_results[bts]['ratio_df']['ratio'].copy()
            ratio_s.name = f'bts_{bts}'
            bts_ratios_df = pd.concat([bts_ratios_df, ratio_s], axis=1)
            # bts_ratios_df[bts]=bts_results[bts]['ratio_df']['ratio']
        bts_ratios_df['ROW_Q025'] = bts_ratios_df.quantile(q=.025, axis=1)
        bts_ratios_df['ROW_Q975'] = bts_ratios_df.quantile(q=.975, axis=1)
        bts_ratios_df['fit_x'] = bts_results[1]['ratio_df']['fit_x']  # Add x

        # Get (daytime) GPP and RECO values at detected CHD threshold

        # Threshold from threshold detection
        _thres = self.results_threshold_detection['thres_chd']
        _thres = np.round(_thres, 4)  # Round to 4 digits to facilitate exact search match

        # Get x values from (daytime) GPP and RECO fitting
        _gpp_results = bts_results[0][self.gpp_col]['fit_df']['fit_x'].values
        _gpp_results = np.round(_gpp_results, 4)
        _reco_results = bts_results[0][self.reco_col]['fit_df']['fit_x'].values
        _reco_results = np.round(_reco_results, 4)

        # Find x threshold in x values used in fitting, returns location index
        _gpp_thres_ix = np.where(_gpp_results == _thres)
        _reco_thres_ix = np.where(_reco_results == _thres)

        # Get values from fit at found location
        _gpp_at_thres = bts_results[0][self.gpp_col]['fit_df']['nom'].iloc[_gpp_thres_ix]
        _reco_at_thres = bts_results[0][self.reco_col]['fit_df']['nom'].iloc[_reco_thres_ix]
        _ratio_at_thres = _gpp_at_thres / _reco_at_thres

        daytime_values_at_threshold = {
            'GPP': _gpp_at_thres,
            'RECO': _reco_at_thres,
            'RATIO': _ratio_at_thres
        }

        # Collect results
        self._results_daytime_analysis = dict(bts_results=bts_results,
                                              bts_ratios_df=bts_ratios_df,
                                              daytime_values_at_threshold=daytime_values_at_threshold)

    def detect_chd_threshold(self):
        """Detect critical heat days x threshold for NEE"""

        # Fit to bootstrapped data, daily time resolution
        # Stored as bootstrap runs > 0 (bts>0)
        bts_results = self._bootstrap_fits(df_aggs=self.df_aggs,
                                           x_col=self.x_col,
                                           x_agg='max',
                                           y_cols=[self.nee_col, self.gpp_col, self.reco_col],
                                           y_agg='sum',
                                           ratio_cols=None,
                                           fit_to_bins=self.usebins,
                                           detect_zerocrossing=True)

        # Get flux equilibrium points (RECO = GPP) from bootstrap runs
        bts_zerocrossings_df = self._bts_zerocrossings_collect(bts_results=bts_results)

        # Calc flux equilibrium points aggregates from bootstrap runs
        bts_zerocrossings_aggs = self._zerocrossings_aggs(bts_zerocrossings_df=bts_zerocrossings_df)

        # Threshold for Critical Heat Days (CHDs)
        # defined as the linecrossing median x (e.g. VPD) from bootstrap runs
        thres_chd = bts_zerocrossings_aggs['x_median']

        # Collect days above or equal to CHD threshold
        df_aggs_chds = self.df_aggs.loc[self.df_aggs[self.x_col]['max'] >= thres_chd, :].copy()

        # Number of days above CHD threshold
        num_chds = len(df_aggs_chds)

        # Collect Near-Critical Heat Days (nCHDs)
        # With the number of CHDs known, collect data for the same number
        # of days below of equal to CHD threshold.
        # For example: if 10 CHDs were found, nCHDs are the 10 days closest
        # to the CHD threshold (below or equal to the threshold).
        sortby_col = (self.x_col, 'max')
        nchds_start_ix = num_chds
        nchds_end_ix = num_chds * 2
        df_aggs_nchds = self.df_aggs \
                            .sort_values(by=sortby_col, ascending=False) \
                            .iloc[nchds_start_ix:nchds_end_ix]

        # Threshold for nCHDs
        # The lower threshold is the minimum of found x maxima
        thres_nchds_lower = df_aggs_nchds[self.x_col]['max'].min()
        thres_nchds_upper = thres_chd

        # Number of days above nCHD threshold and below or equal CHD threshold
        num_nchds = len(df_aggs_nchds)

        # Collect results
        self._results_threshold_detection = dict(
            bts_results=bts_results,
            bts_zerocrossings_df=bts_zerocrossings_df,
            bts_zerocrossings_aggs=bts_zerocrossings_aggs,
            thres_chd=thres_chd,
            thres_nchds_lower=thres_nchds_lower,
            thres_nchds_upper=thres_nchds_upper,
            df_aggs_chds=df_aggs_chds,
            df_aggs_nchds=df_aggs_nchds,
            num_chds=num_chds,
            num_nchds=num_nchds
        )

    # def detect_chd_threshold_from_partitioned(self):
    #
    #     # Fit to bootstrapped data
    #     # Stored as bootstrap runs > 0 (bts>0)
    #     bts_results = self._bootstrap_fits(df=self.df_daytime_aggs,
    #                                        x_agg='max',
    #                                        y_agg='max',
    #                                        fit_to_bins=self.usebins)
    #
    #     # Get flux equilibrium points (RECO = GPP) from bootstrap runs
    #     bts_linecrossings_df = self._bts_linecrossings_collect(bts_results=bts_results)
    #
    #     # Calc flux equilibrium points aggregates from bootstrap runs
    #     bts_linecrossings_aggs = self._linecrossing_aggs(bts_linecrossings_df=bts_linecrossings_df)
    #
    #     # Threshold for Critical Heat Days (CHDs)
    #     # defined as the linecrossing median x (e.g. VPD) from bootstrap runs
    #     thres_chd = bts_linecrossings_aggs['x_median']
    #
    #     # Collect days above or equal to CHD threshold
    #     df_daytime_aggs_chds = self.df_daytime_aggs.loc[self.df_daytime_aggs[self.x_col]['max'] >= thres_chd, :].copy()
    #
    #     # Number of days above CHD threshold
    #     num_chds = len(df_daytime_aggs_chds)
    #
    #     # Collect Near-Critical Heat Days (nCHDs)
    #     # With the number of CHDs known, collect data for the same number
    #     # of days below of equal to CHD threshold.
    #     # For example: if 10 CHDs were found, nCHDs are the 10 days closest
    #     # to the CHD threshold (below or equal to the threshold).
    #     sortby_col = (self.x_col[0], self.x_col[1], 'max')
    #     nchds_start_ix = num_chds
    #     nchds_end_ix = num_chds * 2
    #     df_daytime_aggs_nchds = self.df_daytime_aggs \
    #                                 .sort_values(by=sortby_col, ascending=False) \
    #                                 .iloc[nchds_start_ix:nchds_end_ix]
    #
    #     # Threshold for nCHDs
    #     # The lower threshold is the minimum of found x maxima
    #     thres_nchds_lower = df_daytime_aggs_nchds[self.x_col]['max'].min()
    #     thres_nchds_upper = thres_chd
    #
    #     # Number of days above nCHD threshold and below or equal CHD threshold
    #     num_nchds = len(df_daytime_aggs_nchds)
    #
    #     # Collect results
    #     self._results_threshold_detection = dict(
    #         bts_results=bts_results,
    #         bts_linecrossings_df=bts_linecrossings_df,
    #         bts_linecrossings_aggs=bts_linecrossings_aggs,
    #         thres_chd=thres_chd,
    #         thres_nchds_lower=thres_nchds_lower,
    #         thres_nchds_upper=thres_nchds_upper,
    #         df_daytime_aggs_chds=df_daytime_aggs_chds,
    #         df_daytime_aggs_nchds=df_daytime_aggs_nchds,
    #         num_chds=num_chds,
    #         num_nchds=num_nchds
    #     )

    def find_nee_optimum_range(self):
        # Work w/ daytime data
        opr = FindOptimumRange(df=self.df_daytime, xcol=self.x_col, ycol=self.nee_col,
                               define_optimum='min')
        opr.find_optimum()
        self._results_optimum_range = opr.results_optrange()

    # def analyze_flux(self, flux: str = 'nee'):
    #     """Analyze flux vs CHD threshold"""
    #
    #     thres_chd = self.results_threshold_detection['thres_chd']
    #
    #     fluxcol = self._set_fluxcol(flux=flux)
    #
    #     # Flux subset
    #     flux_df = self.df[[self.x_col, fluxcol]].copy()
    #
    #     # Daily maxima and sums
    #     flux_aggs_df = flux_df.groupby(flux_df.index.date).agg(['max', 'sum'])
    #
    #     # Collect days above the CHD threshold
    #     flux_aggs_chd_df = flux_aggs_df.loc[flux_aggs_df[self.x_col]['max'] >= thres_chd, :]
    #     num_chds = len(flux_aggs_chd_df)
    #
    #     # Collect the same number of days closest to the threshold
    #     flux_aggs_non_chd_df = flux_aggs_df.loc[flux_aggs_df[self.x_col]['max'] < thres_chd, :]
    #     _sortby = (self.x_col[0], self.x_col[1], 'max')
    #     flux_aggs_near_chd_df = flux_aggs_non_chd_df.sort_values(by=_sortby, ascending=False).head(num_chds)
    #     num_near_chds = len(flux_aggs_near_chd_df)
    #     thres_near_chd = flux_aggs_near_chd_df[_sortby].min()
    #
    #     # Fit to bootstrapped data
    #     # Stored as bootstrap runs > 0 (bts>0)
    #     flux_bts_results = self._bootstrap_fits(df_aggs=flux_aggs_df, x_agg='max', y_agg='sum', fit_to_bins=0)
    #
    #     # Collect results
    #     self._results_flux_analysis['bts_results'] = flux_bts_results
    #     self._results_flux_analysis['thres_chd'] = thres_chd  # x threshold for critical heat days
    #     self._results_flux_analysis['thres_near_chd'] = thres_near_chd  # x threshold for near-critical heat days
    #     self._results_flux_analysis['num_chds'] = num_chds  # Number of critical heat days
    #     self._results_flux_analysis['num_near_chds'] = num_near_chds  # Number of near-critical heat days
    #     self._results_flux_analysis['flux_aggs_near_chd_df'] = flux_aggs_near_chd_df  # Near-CHD data
    #     self._results_flux_analysis['flux_aggs_chd_df'] = flux_aggs_chd_df  # CHD data

    def plot_chd_detection_from_nee(self, ax, highlight_year: int = None):
        plot_chd_detection_from_nee(ax=ax, results_chd=self.results_threshold_detection,
                                    y_col=self.nee_col, highlight_year=highlight_year)

    def plot_daytime_analysis(self, ax):
        plot_daytime_analysis(ax=ax,
                              results_chd=self.results_threshold_detection,
                              results_daytime_analysis=self.results_daytime_analysis,
                              gpp_col=self.gpp_col, reco_col=self.reco_col)

    def plot_rolling_bin_aggregates(self, ax):
        """Plot optimum range: rolling bin aggregates"""
        diive.pkgs.analyses.optimumrange.plot_rolling_bin_aggregates(ax=ax, results_optrange=self.results_optimum_range)

    def plot_bin_aggregates(self, ax):
        """Plot optimum range: bin aggregates"""
        diive.pkgs.analyses.optimumrange.plot_bin_aggregates(ax=ax, results_optrange=self.results_optimum_range)

    def plot_vals_in_optimum_range(self, ax):
        """Plot optimum range: values in, above and below optimum per year"""
        diive.pkgs.analyses.optimumrange.plot_vals_in_optimum_range(ax=ax, results_optrange=self.results_optimum_range)

    def _set_fluxcol(self, flux: str = 'nee'):
        fluxcol = None
        if flux == 'nee':
            fluxcol = self.nee_col
        if flux == 'gpp':
            fluxcol = self.gpp_col
        if flux == 'reco':
            fluxcol = self.reco_col
        return fluxcol

    def _resample_dataset(self, df: DataFrame, aggs: list, day_start_hour: int = None):
        """Resample to daily values from *day_start_hour* to *day_start_hour*"""
        df_aggs = df.resample('D', offset=f'{day_start_hour}H').agg(aggs)
        df_aggs = df_aggs.where(df_aggs[self.x_col]['count'] == 48).dropna()  # Full days only
        return df_aggs

    def _prepare_daynight_datasets(self, aggs):
        """Create separate daytime/nighttime datasets and aggregate"""

        # Get daytime data from dataset
        df_daytime, \
        df_nighttime, \
        grp_daynight_col, \
        date_col, \
        flag_daynight_col = \
            self._get_daynight_data()

        args = dict(groupby_col=grp_daynight_col,
                    date_col=date_col,
                    min_vals=0,
                    aggs=aggs)

        # Aggregate daytime dataset
        df_daytime_aggs = self._aggregate_by_group(df=df_daytime, **args)

        # Aggregate nighttime dataset
        df_nighttime_aggs = self._aggregate_by_group(df=df_nighttime, **args)

        # print(len(df_daytime_aggs))
        # print(len(df_nighttime_aggs))

        return df_daytime, df_daytime_aggs, df_nighttime, df_nighttime_aggs

    def _get_daynight_data(self):
        """Get daytime data from dataset"""
        return frames.splitdata_daynight(
            df=self.df.copy(),
            split_on=self.daynight_split,
            split_day_start_hour=self.daynight_split_day_start_hour,
            split_day_end_hour=self.daynight_split_day_end_hour,
            split_threshold=self.daytime_threshold,
            split_flagtrue=self.set_daytime_if
        )

    def _zerocrossings_aggs(self, bts_zerocrossings_df: pd.DataFrame) -> dict:
        """Aggregate linecrossing results from bootstrap runs"""
        # linecrossings_x = []
        # linecrossings_y_gpp = []
        # for b in range(1, self.bootstrap_runs + 1):
        #     linecrossings_x.append(self.bts_results[b]['linecrossing_vals']['x_col'])
        #     linecrossings_y_gpp.append(self.bts_results[b]['linecrossing_vals']['gpp_nom'])

        zerocrossings_aggs = dict(
            x_median=round(bts_zerocrossings_df['x_col'].median(), 6),
            x_min=bts_zerocrossings_df['x_col'].min(),
            x_max=bts_zerocrossings_df['x_col'].max(),
            y_nee_median=bts_zerocrossings_df['nee_nom'].median(),
            y_nee_min=bts_zerocrossings_df['nee_nom'].min(),
            y_nee_max=bts_zerocrossings_df['nee_nom'].max(),
        )

        return zerocrossings_aggs

    def _bts_zerocrossings_collect(self, bts_results):
        bts_linecrossings_df = pd.DataFrame()
        for bts in range(1, self.bootstrap_runs + 1):
            _dict = bts_results[bts]['zerocrossing_vals']
            _series = pd.Series(_dict)
            _series.name = bts
            if bts == 1:
                bts_linecrossings_df = pd.DataFrame(_series).T
            else:
                bts_linecrossings_df = bts_linecrossings_df.append(_series.T)
        return bts_linecrossings_df

    def _bootstrap_fits(self,
                        df_aggs: DataFrame,
                        x_col: str,
                        x_agg: str,
                        y_cols: list,
                        y_agg: str,
                        ratio_cols: list = None,
                        fit_to_bins: int = 10,
                        detect_zerocrossing: bool = False) -> dict:
        """Bootstrap ycols and fit to x"""

        # Get column names in aggregated df
        x_col = (x_col, x_agg)
        _y_agg_cols = []
        for _ycol in y_cols:
            _y_agg_cols.append((_ycol, y_agg))
        y_cols = _y_agg_cols
        bts_results = {}
        bts = 0

        while bts < self.bootstrap_runs + 1:
            print(f"Bootstrap run #{bts}")
            fit_results = {}

            if bts > 0:
                # Bootstrap data
                bts_df = df_aggs.sample(n=int(len(df_aggs)), replace=True, random_state=self.bootstrapping_random_state)
            else:
                # First run (bts=0) is with measured data (not bootstrapped)
                bts_df = df_aggs.copy()

            try:
                for y_col in y_cols:

                    # Fit
                    fitter = BinFitterCP(df=bts_df,
                                         x_col=x_col,
                                         y_col=y_col,
                                         num_predictions=1000,
                                         predict_min_x=self.predict_min_x,
                                         predict_max_x=self.predict_max_x,
                                         bins_x_num=fit_to_bins,
                                         bins_y_agg='mean',
                                         fit_type='quadratic')
                    fitter.run()
                    cur_fit_results = fitter.get_results()

                    # Store fit results for current y
                    fit_results[y_col[0]] = cur_fit_results

                    # Zero crossing for NEE
                    if (y_col[0] == self.nee_col) & detect_zerocrossing:
                        zerocrossing_vals = \
                            self._detect_zerocrossing_nee(fit_results_nee=cur_fit_results['fit_df'])

                        if isinstance(zerocrossing_vals, dict):
                            pass
                        else:
                            raise ValueError

                        fit_results['zerocrossing_vals'] = zerocrossing_vals
                        print(fit_results['zerocrossing_vals'])

                    # import matplotlib.pyplot as plt
                    # fit_results['fit_df'][['fit_x', 'nom']].plot()
                    # plt.scatter(fit_results['fit_df']['fit_x'], fit_results['fit_df']['nom'])
                    # plt.scatter(df[x_col], df[y_col])
                    # plt.show()


            except ValueError:
                print(f"(!) WARNING Bootstrap run #{bts} was not successful, trying again")

            # Ratio
            if ratio_cols:
                ratio_df = pd.DataFrame()
                ratio_df['fit_x'] = fit_results[ratio_cols[0]]['fit_df']['fit_x']
                ratio_df[f'{ratio_cols[0]}_nom'] = fit_results[ratio_cols[0]]['fit_df']['nom']
                ratio_df[f'{ratio_cols[1]}_nom'] = fit_results[ratio_cols[1]]['fit_df']['nom']
                ratio_df['ratio'] = ratio_df[f'{ratio_cols[0]}_nom'].div(ratio_df[f'{ratio_cols[1]}_nom'])
                fit_results['ratio_df'] = ratio_df

            # Store bootstrap results in dict
            bts_results[bts] = fit_results
            bts += 1

        return bts_results

    # def _fits(self, df, x_agg: str, y_agg: str, fit_to_bins: int):
    #     """Make fit to GPP, RECO and NEE vs x"""
    #
    #     nee_fit_results = None
    #     linecrossing_vals = None
    #
    #     print(f"    Fitting {self.nee_col}")
    #     nee_fit_results = \
    #         self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
    #                        y_col=self.nee_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
    #     # fitplot(x=nee_fit_results['x'], y=nee_fit_results['y'], fit_df=nee_fit_results['fit_df'])
    #
    #     # Line crossings
    #     linecrossing_vals = \
    #         self._detect_zerocrossing_nee(fit_results_nee=gpp_fit_results,
    #                                       reco_fit_results=reco_fit_results)
    #     if isinstance(linecrossing_vals, pd.Series):
    #         pass
    #     else:
    #         raise ValueError
    #
    #     # Store bootstrap results in dict
    #     bts_results = {'nee': nee_fit_results,
    #                    'linecrossing_vals': linecrossing_vals}
    #
    #     return bts_results

    # def _fits_gppreco(self, df, x_agg: str, y_agg: str, fit_to_bins: int):
    #     """Make fit to GPP, RECO and NEE vs x"""
    #
    #     # Check what is available
    #     gpp = True if self.gpp_col in df else False
    #     reco = True if self.reco_col in df else False
    #     nee = True if self.nee_col in df else False
    #     gpp_fit_results = None
    #     reco_fit_results = None
    #     nee_fit_results = None
    #     linecrossing_vals = None
    #
    #     # GPP sums (class mean) vs classes of x (max)
    #     if gpp:
    #         print(f"    Fitting {self.gpp_col}")
    #         gpp_fit_results = \
    #             self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
    #                            y_col=self.gpp_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
    #         # fitplot(x=gpp_fit_results['x'], y=gpp_fit_results['y'], fit_df=gpp_fit_results['fit_df'])
    #
    #     # RECO sums (class mean) vs classes of x (max)
    #     if reco:
    #         print(f"    Fitting {self.reco_col}")
    #         reco_fit_results = \
    #             self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
    #                            y_col=self.reco_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
    #         # fitplot(x=reco_fit_results['x'], y=reco_fit_results['y'], fit_df=reco_fit_results['fit_df'])
    #
    #     # NEE sums (class mean) vs classes of x (max)
    #     if nee:
    #         print(f"    Fitting {self.nee_col}")
    #         nee_fit_results = \
    #             self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
    #                            y_col=self.nee_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
    #         # fitplot(x=nee_fit_results['x'], y=nee_fit_results['y'], fit_df=nee_fit_results['fit_df'])
    #
    #     # Line crossings
    #     if gpp and reco:
    #         linecrossing_vals = \
    #             self._detect_linecrossing(gpp_fit_results=gpp_fit_results,
    #                                       reco_fit_results=reco_fit_results)
    #         if isinstance(linecrossing_vals, pd.Series):
    #             pass
    #         else:
    #             raise ValueError
    #
    #     # Store bootstrap results in dict
    #     bts_results = {'gpp': gpp_fit_results,
    #                    'reco': reco_fit_results,
    #                    'nee': nee_fit_results,
    #                    'linecrossing_vals': linecrossing_vals}
    #
    #     return bts_results

    def _detect_zerocrossing_nee(self, fit_results_nee: dict):
        # kudos: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value

        num_zerocrossings = None
        zerocrossings_ix = None

        # Collect predicted vals in df
        zerocrossings_df = pd.DataFrame()
        zerocrossings_df['x_col'] = fit_results_nee['fit_x']
        zerocrossings_df['nee_nom'] = fit_results_nee['nom']

        # Check values above/below zero
        _signs = np.sign(zerocrossings_df['nee_nom'])
        _signs_max = _signs.max()
        _signs_min = _signs.min()
        _signs_num_totalvals = len(_signs)
        _signs_num_abovezero = _signs.loc[_signs > 0].count()
        _signs_num_belowzero = _signs.loc[_signs < 0].count()

        if _signs_max == _signs_min:
            print("NEE does not cross zero-line.")
        else:
            zerocrossings_ix = np.argwhere(np.diff(_signs)).flatten()
            num_zerocrossings = len(zerocrossings_ix)

        # linecrossings_idx = \
        #     np.argwhere(np.diff(np.sign(zerocrossings_df['gpp_nom'] - zerocrossings_df['reco_nom']))).flatten()
        # num_linecrossings = len(linecrossings_idx)

        # There must be one single line crossing to accept result
        # If there is more than one line crossing, reject result
        if num_zerocrossings > 1:
            return None

        # Values at zero crossing needed
        # Found index is last element *before* zero-crossing, therefore + 1
        zerocrossing_vals = zerocrossings_df.iloc[zerocrossings_ix[0] + 1]
        zerocrossing_vals = zerocrossing_vals.to_dict()

        # NEE at zero crossing must zero or above,
        # i.e. reject if NEE after zero-crossing does not change to emission
        if (zerocrossing_vals['nee_nom'] < 0):
            return None

        # x value must be above threshold to be somewhat meaningful, otherwise reject result
        if (zerocrossing_vals['x_col'] < 10):
            # VPD is too low, must be at least 10 for valid crossing
            return None

        return zerocrossing_vals

    # def _detect_linecrossing(self, gpp_fit_results, reco_fit_results):
    #     # Collect predicted vals in df
    #     linecrossings_df = pd.DataFrame()
    #     linecrossings_df['x_col'] = gpp_fit_results['fit_df']['fit_x']
    #     linecrossings_df['gpp_nom'] = gpp_fit_results['fit_df']['nom']
    #     linecrossings_df['reco_nom'] = reco_fit_results['fit_df']['nom']
    #
    #     # https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
    #     linecrossings_idx = \
    #         np.argwhere(np.diff(np.sign(linecrossings_df['gpp_nom'] - linecrossings_df['reco_nom']))).flatten()
    #
    #     num_linecrossings = len(linecrossings_idx)
    #
    #     # There must be one single line crossing to accept result
    #     if num_linecrossings == 1:
    #
    #         # Flux values at line crossing
    #         linecrossing_vals = linecrossings_df.iloc[linecrossings_idx[0] + 1]
    #
    #         # GPP and RECO must be positive, also x value must be
    #         # above threshold, otherwise reject result
    #         if (linecrossing_vals['gpp_nom'] < 0) \
    #                 | (linecrossing_vals['reco_nom'] < 0) \
    #                 | (linecrossing_vals['x_col'] < 5):
    #             return None
    #
    #         return linecrossing_vals
    #
    #     else:
    #         # If there is more than one line crossing, reject result
    #         return None

    # def _calc_fit(self, df, x_col, x_agg, y_col, y_agg, fit_to_bins):
    #     """Call BinFitterCP and fit to x and y"""
    #
    #     # Names of x and y cols in aggregated df
    #     x_col = (x_col[0], x_col[1], x_agg)
    #     y_col = (y_col[0], y_col[1], y_agg)
    #
    #     fitter = BinFitterCP(df=df,
    #                          x_col=x_col,
    #                          y_col=y_col,
    #                          num_predictions=1000,
    #                          predict_min_x=self.predict_min_x,
    #                          predict_max_x=self.predict_max_x,
    #                          bins_x_num=fit_to_bins,
    #                          bins_y_agg='median',
    #                          fit_type='quadratic')
    #     fitter.run()
    #     return fitter.get_results()

    def _aggregate_by_group(self, df, groupby_col, date_col, min_vals, aggs: list) -> pd.DataFrame:
        """Aggregate dataset by *day/night groups*"""

        # Aggregate values by day/night group membership, this drops the date col
        agg_df = \
            df.groupby(groupby_col) \
                .agg(aggs)
        # .agg(['median', q25, q75, 'count', 'max'])
        # .agg(['median', q25, q75, 'min', 'max', 'count', 'mean', 'std', 'sum'])

        # Add the date col back to data
        grp_daynight_col = groupby_col
        agg_df[grp_daynight_col] = agg_df.index

        # For each day/night group, detect its start and end time

        ## Start date (w/ .idxmin)
        grp_starts = df.groupby(groupby_col).idxmin()[date_col].dt.date
        grp_starts = grp_starts.to_dict()
        grp_startdate_col = '.GRP_STARTDATE'
        agg_df[grp_startdate_col] = agg_df[grp_daynight_col].map(grp_starts)

        ## End date (w/ .idxmax)
        grp_ends = df.groupby(groupby_col).idxmax()[date_col].dt.date
        grp_ends = grp_ends.to_dict()
        grp_enddate_col = '.GRP_ENDDATE'
        agg_df[grp_enddate_col] = agg_df[grp_daynight_col].map(grp_ends)

        # Set start date as index
        agg_df = agg_df.set_index(grp_startdate_col)
        agg_df.index = pd.to_datetime(agg_df.index)

        # Keep consecutive time periods with enough values (min. 11 half-hours)
        agg_df = agg_df.where(agg_df[self.x_col]['count'] >= min_vals).dropna()

        return agg_df


def plot_daytime_analysis(ax,
                          results_chd,
                          results_daytime_analysis,
                          gpp_col: str,
                          reco_col: str):
    """Plot daytime fluxes"""

    # fig = plt.figure(figsize=(9, 9))
    # gs = gridspec.GridSpec(1, 1)  # rows, cols
    # # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    # ax = fig.add_subplot(gs[0, 0])

    # Get data
    bts_results = results_chd['bts_results'][0]  # 0 means non-bootstrapped data
    bts_zerocrossings_aggs = results_chd['bts_zerocrossings_aggs']
    bts_results_daytime = results_daytime_analysis['bts_results'][0]  # 0 means non-bootstrapped data

    # Ratio GPP:RECO and uncertainty
    line_ratio, = ax.plot(bts_results_daytime['ratio_df']['fit_x'],
                          bts_results_daytime['ratio_df']['ratio'],
                          c='black', lw=5, zorder=99, alpha=1, label="ratio GPP:RECO")
    line_ratio95 = ax.fill_between(results_daytime_analysis['bts_ratios_df']['fit_x'],
                                   results_daytime_analysis['bts_ratios_df']['ROW_Q025'],
                                   results_daytime_analysis['bts_ratios_df']['ROW_Q975'],
                                   alpha=.2, color='black', zorder=99,
                                   label="95% confidence region")

    # Vertical line showing median CHD threshold from bootstrap runs
    line_chd_vertical = ax.axvline(bts_zerocrossings_aggs['x_median'], lw=2, color='#2196F3', ls='--',
                                   label=f"critical heat days threshold, VPD = {bts_zerocrossings_aggs['x_median']:.1f} hPa")

    # Rectangle (bootstrapped results)
    num_bootstrap_runs = len(results_chd['bts_results'].keys()) - 1  # -1 b/c zero run is non-bootstrapped
    range_bts_netzeroflux = rectangle(ax=ax,
                                      rect_lower_left_x=bts_zerocrossings_aggs['x_min'],
                                      rect_lower_left_y=0,
                                      rect_width=bts_zerocrossings_aggs['x_max'] - bts_zerocrossings_aggs['x_min'],
                                      rect_height=1,
                                      label=f"threshold range ({num_bootstrap_runs} bootstraps)",
                                      color='#2196F3')

    ax_twin = ax.twinx()

    # GPP
    _numvals_per_bin = bts_results_daytime[gpp_col]['numvals_per_bin']
    flux_bts_results = bts_results_daytime[gpp_col]
    line_xy_gpp, line_fit_gpp, line_fit_ci_gpp, line_fit_pb_gpp = \
        fitplot(ax=ax_twin,
                label=gpp_col,
                # highlight_year=highlight_year,
                flux_bts_results=flux_bts_results, alpha=.2,
                edgecolor='#4CAF50', color='none', color_fitline='#4CAF50')  # green 500

    # RECO
    _numvals_per_bin = bts_results_daytime[reco_col]['numvals_per_bin']
    flux_bts_results = bts_results_daytime[reco_col]
    line_xy_reco, line_fit_reco, line_fit_ci_reco, line_fit_pb_reco = \
        fitplot(ax=ax_twin,
                label=reco_col,
                # highlight_year=highlight_year,
                flux_bts_results=flux_bts_results, alpha=.2, marker='s',
                edgecolor='#9C27B0', color='none', color_fitline='#9C27B0')  # purple 500

    # Values at threshold
    scatter_gpp_at_thres = ax_twin.scatter(results_chd['thres_chd'],
                                           results_daytime_analysis['daytime_values_at_threshold']['GPP'],
                                           edgecolor='#2196F3', color='none', alpha=1, s=250, lw=2,
                                           label='xxx', zorder=99, marker='s')

    scatter_reco_at_thres = ax_twin.scatter(results_chd['thres_chd'],
                                            results_daytime_analysis['daytime_values_at_threshold']['RECO'],
                                            edgecolor='#2196F3', color='none', alpha=1, s=250, lw=2,
                                            label='xxx', zorder=99, marker='s')
    scatter_ratio_at_thres = ax.scatter(results_chd['thres_chd'],
                                        results_daytime_analysis['daytime_values_at_threshold']['RATIO'],
                                        edgecolor='#2196F3', color='none', alpha=1, s=250, lw=2,
                                        label='xxx', zorder=99, marker='s')

    _ratio_at_thres = results_daytime_analysis['daytime_values_at_threshold']['RATIO'].values
    ax.text(results_chd['thres_chd'], _ratio_at_thres,
            f"    Ratio at threshold: {_ratio_at_thres}",
            size=theme.FONTSIZE_ANNOTATIONS_SMALL,
            color='#2196F3', backgroundcolor='none',
            alpha=1, horizontalalignment='left', verticalalignment='center',
            bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'))

    l = ax.legend(
        [
            line_ratio,
            line_xy_gpp,
            line_xy_reco,
            (line_fit_gpp, line_fit_reco),
            (line_fit_ci_gpp, line_fit_ci_reco),
            (line_fit_pb_gpp, line_fit_pb_reco),
            line_chd_vertical,
            range_bts_netzeroflux
        ],
        [
            line_ratio.get_label(),
            line_xy_gpp.get_label(),
            line_xy_reco.get_label(),
            line_fit_gpp.get_label(),
            line_fit_ci_gpp.get_label(),
            line_fit_pb_gpp.get_label(),
            line_chd_vertical.get_label(),
            range_bts_netzeroflux.get_label()
        ],
        bbox_to_anchor=(0, 1),
        loc="lower left",
        ncol=2,
        scatterpoints=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)})

    # Format
    # ax.axhline(0, lw=1, color='black')
    xlabel = "daily maximum VPD ($hPa$)"
    ylabel = r"GPP : RECO ($ratio$)"
    # ylabel = r"daily cumulative carbon flux ($\mu mol \/\ CO_2 \/\ m^{-2} \/\ s^{-1}$)"
    plotfuncs.default_format(ax=ax, txt_xlabel=xlabel, txt_ylabel=ylabel,
                             ticks_width=1)

    # fig.show()


def plot_chd_detection_from_nee(ax, results_chd: dict, y_col: str, highlight_year: int = None):
    """Plot results from critical heat days threshold detection"""

    bts_results = results_chd['bts_results'][0]  # 0 means non-bootstrapped data
    bts_zerocrossings_aggs = results_chd['bts_zerocrossings_aggs']

    # NEE
    _numvals_per_bin = bts_results[y_col]['numvals_per_bin']
    flux_bts_results = bts_results[y_col]
    line_xy_nee, line_fit_nee, line_fit_ci_nee, line_fit_pb_nee, line_highlight = \
        fitplot(ax=ax,
                label="NEE",
                # label=f"NEE ({_numvals_per_bin['min']:.0f} - {_numvals_per_bin['max']:.0f} values per bin)",
                highlight_year=highlight_year,
                flux_bts_results=flux_bts_results,
                edgecolor='#B0BEC5', color='none', color_fitline='#E53935')  # color red 600

    # # Actual non-bootstrapped line crossing, the point where RECO = GPP
    # line_netzeroflux = ax.scatter(bts_results['zerocrossing_vals']['x_col'],
    #                               bts_results['zerocrossing_vals']['nee_nom'],
    #                               edgecolor='black', color='none', alpha=1, s=90, lw=2,
    #                               label='net zero flux', zorder=99, marker='s')

    # Bootstrapped line crossing, the point where RECO = GPP
    line_bts_median_netzeroflux = ax.scatter(bts_zerocrossings_aggs['x_median'],
                                             0,
                                             edgecolor='#2196F3', color='none', alpha=1, s=250, lw=2,
                                             label='net zero flux (bootstrapped median)', zorder=99, marker='s')

    # Vertical line showing median CHD threshold from bootstrap runs
    line_chd_vertical = ax.axvline(bts_zerocrossings_aggs['x_median'], lw=2, color='#2196F3', ls='--',
                                   label=f"critical heat days threshold, VPD = {bts_zerocrossings_aggs['x_median']:.1f} hPa")

    # Rectangle bootstrap range
    num_bootstrap_runs = len(results_chd['bts_results'].keys()) - 1  # -1 b/c zero run is non-bootstrapped
    range_bts_netzeroflux = rectangle(ax=ax,
                                      rect_lower_left_x=bts_zerocrossings_aggs['x_min'],
                                      rect_lower_left_y=0,
                                      rect_width=bts_zerocrossings_aggs['x_max'] - bts_zerocrossings_aggs['x_min'],
                                      rect_height=1,
                                      label=f"threshold range ({num_bootstrap_runs} bootstraps)",
                                      color='#2196F3')

    # CHDs
    num_chds = len(results_chd['df_aggs_chds'])
    # area_chd = ax.fill_between([bts_zerocrossings_aggs['x_median'],
    #                             flux_bts_results['fit_df']['fit_x'].max()],
    #                            0, 1,
    #                            color='#BA68C8', alpha=0.1, transform=ax.get_xaxis_transform(),
    #                            label=f"CHDs ({num_chds} days)", zorder=1)
    sym_max = r'$\rightarrow$'
    _pos = bts_zerocrossings_aggs['x_median'] + bts_zerocrossings_aggs['x_max'] * 0.05
    ax.text(_pos, 0.05, f"{num_chds} critical heat days {sym_max}", size=theme.AXLABELS_FONTSIZE,
            color='black', backgroundcolor='none', transform=ax.get_xaxis_transform(),
            alpha=1, horizontalalignment='left', verticalalignment='top')

    # # Near-CHDs
    # line_near_thres_chd = ax.axvline(results_flux['thres_near_chd'],
    #                                  color='black', lw=1, ls='--', zorder=99, label="near-CHD threshold")
    # area_near_chd = ax.fill_between([results_flux['thres_near_chd'],
    #                                  results_flux['thres_chd']],
    #                                 0, 1,
    #                                 color='#FFA726', alpha=0.1, transform=ax.get_xaxis_transform(),
    #                                 label=f"near-CHDs ({num_chds} days)", zorder=1)
    # _pos = bts_zerocrossings_aggs['x_median'] - bts_zerocrossings_aggs['x_max'] * 0.05
    # ax.text(_pos, 0.05, "near-critical heat days",size=theme.FONTSIZE_LABELS_AXIS,
    #         color='black', backgroundcolor='none', transform=ax.get_xaxis_transform(),
    #         alpha=1, horizontalalignment='right', verticalalignment='top')

    # Format
    ax.axhline(0, lw=1, color='black')
    xlabel = "daily maximum VPD ($hPa$)"

    ylabel = r"daily cumulative NEE ($\mu mol \/\ CO_2 \/\ m^{-2} \/\ s^{-1}$)"
    plotfuncs.default_format(ax=ax, txt_xlabel=xlabel, txt_ylabel=ylabel)
    # xlim_lower = flux_bts_results['fit_df']['fit_x'].min()
    # ax.set_xlim([-1, flux_bts_results['fit_df']['fit_x'].max()])
    # ax.set_ylim([bts_results['zerocrossing_vals']['nee_nom'].min(),
    #              bts_results['zerocrossing_vals']['nee_nom'].max()])

    # Custom legend
    # Assign two of the handles to the same legend entry by putting them in a tuple
    # and using a generic handler map (which would be used for any additional
    # tuples of handles like (p1, p3)).
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    l = ax.legend(
        [
            line_xy_nee,
            line_highlight,
            line_fit_nee,
            # (line_fit_gpp, line_fit_reco),  # to display two patches next to each other in same line
            line_fit_ci_nee,
            line_fit_pb_nee,
            line_bts_median_netzeroflux,
            line_chd_vertical,
            range_bts_netzeroflux
        ],
        [
            line_xy_nee.get_label(),
            line_highlight.get_label(),
            line_fit_nee.get_label(),
            line_fit_ci_nee.get_label(),
            line_fit_pb_nee.get_label(),
            line_bts_median_netzeroflux.get_label(),
            line_chd_vertical.get_label(),
            range_bts_netzeroflux.get_label()
        ],
        bbox_to_anchor=(0, 1),
        loc="lower left",
        ncol=2,
        scatterpoints=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)})

    # # Errorbar for bootstrapped min/max threshold findings
    # xerr_upper = bts_zerocrossings_aggs['x_max'] - bts_results['zerocrossing_vals']['x_col']
    # xerr_lower = bts_results['zerocrossing_vals']['x_col'] - bts_zerocrossings_aggs['x_min']
    # line_xerr=plt.errorbar(bts_results['zerocrossing_vals']['x_col'],
    #              bts_results['zerocrossing_vals']['nee_nom'],
    #              xerr=([xerr_lower], [xerr_upper]),
    #              capsize=0, ls='none', color='black',
    #              alpha=.2, elinewidth=10, label="net zero flux range (bootstrapped)")


# def plot_gpp_reco_vs_vpd(ax, results_chd: dict):
#     """Create figure showing GPP and RECO vs VPD"""
#
#     bts_results = results_chd['bts_results'][0]  # 0 means non-bootstrapped data
#     bts_linecrossings_aggs = results_chd['bts_linecrossings_aggs']
#
#     # GPP
#     _numvals_per_bin = bts_results['gpp']['numvals_per_bin']
#     flux_bts_results = bts_results['gpp']
#     line_xy_gpp, line_fit_gpp, line_fit_ci_gpp, line_fit_pb_gpp = \
#         fitplot(ax=ax,
#                 # label=f"GPP ({_numvals_per_bin['min']:.0f} - {_numvals_per_bin['max']:.0f} values per bin)",
#                 flux_bts_results=flux_bts_results,
#                 color='#2196F3',
#                 color_fitline='#2196F3')  # color blue 500
#
#     # RECO
#     _numvals_per_bin = bts_results['reco']['numvals_per_bin']
#     flux_bts_results = bts_results['reco']
#     line_xy_reco, line_fit_reco, line_fit_ci_reco, line_fit_pb_reco = \
#         fitplot(ax=ax,
#                 # label=f"RECO ({_numvals_per_bin['min']:.0f} - {_numvals_per_bin['max']:.0f} values per bin)",
#                 flux_bts_results=flux_bts_results,
#                 color='#E53935',
#                 color_fitline='#E53935')  # color red 600
#
#     # Actual non-bootstrapped line crossing, the point where RECO = GPP
#     line_equilibrium = ax.scatter(bts_results['linecrossing_vals']['x_col'],
#                                   bts_results['linecrossing_vals']['gpp_nom'],
#                                   edgecolor='none', color='black', alpha=1, s=90,
#                                   label='flux equilibrium, RECO = GPP', zorder=99, marker='s')
#
#     # Bootstrapped line crossing, the point where RECO = GPP
#     line_bts_equilibrium = ax.scatter(bts_linecrossings_aggs['x_median'],
#                                       bts_linecrossings_aggs['y_gpp_median'],
#                                       edgecolor='black', color='none', alpha=1, s=90,
#                                       label='flux equilibrium (bootstrapped median)', zorder=99, marker='s')
#
#     # Add rectangle (bootstrapped results)
#     range_bts_equilibrium = rectangle(ax=ax,
#                                       rect_lower_left_x=bts_linecrossings_aggs['x_min'],
#                                       rect_lower_left_y=bts_linecrossings_aggs['y_gpp_min'],
#                                       rect_width=bts_linecrossings_aggs['x_max'] - bts_linecrossings_aggs['x_min'],
#                                       rect_height=bts_linecrossings_aggs['y_gpp_max'] - bts_linecrossings_aggs[
#                                           'y_gpp_min'],
#                                       label="equlibrium range (bootstrapped)")
#
#     # Format
#     xlabel = "classes of daily VPD maxima ($hPa$)"
#     ylabel = r"daytime median flux ($\mu mol \/\ CO_2 \/\ m^{-2} \/\ s^{-1}$)"
#     # ylabel = "daytime median flux" + " (gC $\mathregular{m^{-2} \ d^{-1}}$  --> UNITS ???)"
#     plotfuncs.default_format(ax=ax, txt_xlabel=xlabel, txt_ylabel=ylabel,
#                              fontsize=12, width=1)
#
#     # Custom legend
#     # Assign two of the handles to the same legend entry by putting them in a tuple
#     # and using a generic handler map (which would be used for any additional
#     # tuples of handles like (p1, p3)).
#     # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
#     l = ax.legend(
#         [
#             line_xy_gpp,
#             line_xy_reco,
#             (line_fit_gpp, line_fit_reco),
#             (line_fit_ci_gpp, line_fit_ci_reco),
#             (line_fit_pb_gpp, line_fit_pb_reco),
#             line_equilibrium,
#             line_bts_equilibrium,
#             range_bts_equilibrium
#         ],
#         [
#             line_xy_gpp.get_label(),
#             line_xy_reco.get_label(),
#             line_fit_gpp.get_label(),
#             line_fit_ci_gpp.get_label(),
#             line_fit_pb_gpp.get_label(),
#             line_equilibrium.get_label(),
#             line_bts_equilibrium.get_label(),
#             range_bts_equilibrium.get_label()
#         ],
#         scatterpoints=1,
#         numpoints=1,
#         handler_map={tuple: HandlerTuple(ndivide=None)})


if __name__ == '__main__':
    from tests.testdata.loadtestdata import loadtestdata

    # Load data
    data = loadtestdata()
    df = data['df']
    # from diive.common.dfun.frames import flatten_multiindex_all_df_cols
    # df = flatten_multiindex_all_df_cols(df=df, keep_first_row_only=True)

    # # Use data from May to Sep only
    # maysep_filter = (df.index.month >= 5) & (df.index.month <= 9)
    # df = df.loc[maysep_filter].copy()

    # Settings
    x_col = 'VPD_f'
    nee_col = 'NEE_CUT_f'
    gpp_col = 'GPP_DT_CUT'
    reco_col = 'Reco_DT_CUT'
    # daytime_col = 'Rg_f'

    # Critical heat days
    chd = CriticalHeatDays(
        df=df,
        x_col=x_col,
        nee_col=nee_col,
        gpp_col=gpp_col,
        reco_col=reco_col,
        daynight_split='timestamp',
        daytime_threshold=50,
        set_daytime_if='Larger Than Threshold',
        usebins=0,
        bootstrap_runs=3,
        bootstrapping_random_state=None
    )

    # Critical heat days
    chd.detect_chd_threshold()
    # results = chd.results_threshold_detection()

    # Analyze flux
    chd.analyze_daytime()
    # results = chd.results_daytime_analysis()

    # Plot
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2)  # rows, cols
    gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    chd.plot_chd_detection_from_nee(ax=ax, highlight_year=2019)
    chd.plot_daytime_analysis(ax=ax2)
    fig.tight_layout()
    fig.show()

    # Optimum range
    chd.find_nee_optimum_range()
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 16))
    gs = gridspec.GridSpec(3, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    chd.plot_rolling_bin_aggregates(ax=ax1)
    chd.plot_bin_aggregates(ax=ax2)
    chd.plot_vals_in_optimum_range(ax=ax3)
    fig.show()

    print("END")
