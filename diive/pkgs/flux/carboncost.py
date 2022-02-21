import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# from diive.common.dfun.frames import insert_aggregated_in_hires
import numpy as np
from matplotlib import dates as mdates
from pandas import DataFrame

import diive.core.dfun.frames as frames
# from diive.common.dfun.frames import insert_aggregated_in_hires
from diive.pkgs.flux.criticalheatdays import CriticalHeatDays
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS


class CarbonCost:

    def __init__(
            self,
            df: DataFrame,
            x_col: str,
            nee_col: str,
            gpp_col: str,
            reco_col: str,
            ta_col: str,
            swin_col: str,
            focus_months_from:int=5,
            focus_months_to:int=9
    ):
        """

        Args:
            df:
            x_col:
            nee_col:
            gpp_col:
            reco_col:
        """
        # Full dataset
        self.full_df = df.copy()

        # Focus dataset, e.g. between May and Sep
        focus_months_ix = (df.index.month >= focus_months_from) & (df.index.month <= focus_months_to)
        self.df = df.loc[focus_months_ix, :].dropna()

        self.x_col = x_col
        self.nee_col = nee_col
        self.gpp_col = gpp_col
        self.reco_col = reco_col
        self.ta_col = ta_col
        self.swin_col = swin_col

        # Results from critical heat days analyses
        self._results_chd_threshold_detection = None
        self._results_chd_flux_analysis = None
        self._results_chd_optimum_range = None
        self._chd_instance = None

        # Results from gapfilling
        self._gapfilled_df = None
        self._gf_results = None
        self._carboncost_df = None

    @property
    def carboncost_df(self) -> DataFrame:
        """Return gap-filled dataframe"""
        if not isinstance(self._carboncost_df, DataFrame):
            raise Exception('No gap-filled data found')
        return self._carboncost_df

    @property
    def gapfilled_df(self) -> DataFrame:
        """Return gap-filled dataframe"""
        if not isinstance(self._gapfilled_df, DataFrame):
            raise Exception('No gap-filled data found')
        return self._gapfilled_df

    @property
    def gf_results(self) -> DataFrame:
        """Return gap-filled data results"""
        if not self._gf_results:
            raise Exception('No gap-filled data found')
        return self._gf_results

    @property
    def results_chd_threshold_detection(self) -> dict:
        """Return bootstrap results for daily flux"""
        if not self._results_chd_threshold_detection:
            raise Exception('Results for CriticalHeatDays are empty')
        return self._results_chd_threshold_detection

    @property
    def results_chd_flux_analysis(self) -> dict:
        """Return bootstrap results for flux analysis"""
        if not self._results_chd_flux_analysis:
            raise Exception('Results for CriticalHeatDays are empty')
        return self._results_chd_flux_analysis

    @property
    def results_chd_optimum_range(self) -> dict:
        """Return results for optimum range"""
        if not self._results_chd_optimum_range:
            raise Exception('Results for CriticalHeatDays are empty')
        return self._results_chd_optimum_range

    @property
    def chd_instance(self):
        """Return results for critical heat days"""
        if not self._chd_instance:
            raise Exception('No instance for CriticalHeatDays found')
        return self._chd_instance

    def detect_criticalheatdays(self, **kwargs):
        self._chd_instance = self._detect_criticalheatdays(**kwargs)

    def gapfill_criticalheatdays(self):
        self._carboncost_df, self._gapfilled_df, self._gf_results = self._gapfill_criticalheatdays()


    def _get_thresholds(self):
        # # Get thresholds that define near-critical heat days (CHDs)
        # thres_chds = self.results_chd_threshold_detection['thres_chds']
        # thres_nchds_lower = self.results_chd_threshold_detection['thres_nchds_lower']
        # For testing only:
        thres_chds = 20  # For testing only
        thres_nchds_lower = 15  # For testing only
        return thres_chds, thres_nchds_lower

    def _gapfill_criticalheatdays(self):
        # Get thresholds for CHDs and lower threshold for nCHDs
        thres_chds, thres_nchds_lower = self._get_thresholds()

        # Limit/remove CHD data
        df, ta_limited_col, vpd_limited_col, nee_limited_col = \
            self._limit_chd_data(thres_chds=thres_chds, thres_nchds_lower=thres_nchds_lower)

        # Make subset with vars required for gapfilling
        df_limited = df[[nee_limited_col, ta_limited_col, vpd_limited_col, self.swin_col]].copy()

        # # TODO code for pairing hires daytime with its previous nighttime means
        # # Add flag for day/night groups (nights are consecutive, i.e. from 19:00 to 07:00)
        # df, _, _, grp_daynight_col, date_col, flag_daynight_col = \
        #     frames.splitdata_daynight(df=df_limited, split_on='timestamp',
        #                               split_day_start_hour=7, split_day_end_hour=18)
        #
        # df_grpaggs = df.groupby(grp_daynight_col).mean()
        # df_grpaggs[grp_daynight_col] = df_grpaggs.index  # Needed for merging
        # df_grpaggs.index.name = 'INDEX'  # To avoid duplicate names
        # df_grpaggs.columns = [f"{col}_GRP_MEAN" for col in df_grpaggs.columns]
        # keepcols = ['TA_LIMITED_GRP_MEAN', 'VPD_LIMITED_GRP_MEAN', '.GRP_DAYNIGHT_GRP_MEAN']
        # df_grpaggs = df_grpaggs[keepcols].copy()
        #
        # # Means from previous day/night grp
        # # This pairs each day or night group with the mean of the previous group
        # # Daytime (07:00-19:00) is paired with the means of the *previous* nighttime (19:00-07:00)
        # # Nighttime is paired with the means of the *previous* daytime
        # df_grpaggs_prevgrp = df_grpaggs.copy()
        # # Shift grouping col by one record for merging each record with the mean of the respective previous record
        # # e.g. daytime (grp 1324) is paired with the mean of the previous nighttime (grp 1323)
        # df_grpaggs_prevgrp['.PAIRWITH_GRP'] = df_grpaggs_prevgrp['.GRP_DAYNIGHT_GRP_MEAN'].shift(-1)
        # df_grpaggs_prevgrp.columns = [f"{col}_PREVGRP" for col in df_grpaggs_prevgrp.columns]
        #
        # df['TIMESTAMP'] = df.index  # Store before merging
        #
        # # Merge data with grp means
        # df = df.merge(df_grpaggs, how='left',
        #               left_on=grp_daynight_col,
        #               right_on='.GRP_DAYNIGHT_GRP_MEAN')
        #
        # # Merge data with means of previous grp
        # df = df.merge(df_grpaggs_prevgrp, how='left',
        #               left_on=grp_daynight_col,
        #               right_on='.PAIRWITH_GRP_PREVGRP')
        #
        # df.set_index('TIMESTAMP', inplace=True)
        #
        # keepcols = ['NEE_LIMITED', 'TA_LIMITED', 'VPD_LIMITED', 'Rg_f',
        #       'TA_LIMITED_GRP_MEAN', 'VPD_LIMITED_GRP_MEAN',
        #       'TA_LIMITED_GRP_MEAN_PREVGRP', 'VPD_LIMITED_GRP_MEAN_PREVGRP']
        #
        # df = df[keepcols].copy()
        #
        # df = df.dropna()  # todo currently the first night is lost due to shift

        # df.to_csv('M:\Downloads\_temp2\check.csv')

        # Gapfilling
        rfts = RandomForestTS(df=df_limited,
                              target_col=nee_limited_col,
                              verbose=1,
                              random_state=42,
                              n_estimators=3,
                              bootstrap=True)
        rfts.run()

        # Get gapfilled focus range
        _gapfilled_df, _gf_results = rfts.get_gapfilled_dataset()

        # Reindex to have same index as full dataset (full timestamp range)
        _gapfilled_df = _gapfilled_df.reindex(self.full_df.index)

        _carboncost_df = self.full_df.copy()
        _carboncost_df['NEE_LIMITED_gfRF'] = _gapfilled_df['NEE_LIMITED_gfRF'].fillna(self.full_df[self.nee_col])
        _carboncost_df['QCF_NEE_LIMITED_gfRF'] = _gapfilled_df['QCF_NEE_LIMITED_gfRF']
        _carboncost_df['DIFF'] = _carboncost_df['NEE_LIMITED_gfRF'].sub(_carboncost_df[self.nee_col])
        _carboncost_df['DIFF_CUMSUM'] = _carboncost_df['DIFF'].cumsum()

        return _carboncost_df, _gapfilled_df, _gf_results

        # import matplotlib.pyplot as plt
        # import pandas as pd
        # plot_df = self.gapfilled_df[['NEE_LIMITED_gfRF', 'NEE_LIMITED']].copy()
        # plot_df = pd.concat([plot_df, self.df[nee_col]], axis=1)
        # plot_df.plot(xlim=('2019-06-15', '2019-07-15'), alpha=.5)
        # plot_df.cumsum().plot(alpha=.5)
        # plt.show()

    def _limit_chd_data(self, thres_nchds_lower: float, thres_chds: float) -> tuple[DataFrame, str, str, str]:
        """Limit/remove data on critical heat days

        - Set CHD data for TA and VPD to their diel cycle medians
        - Remove NEE CHD data
        """

        # Insert aggregated x as column in hires dataframe
        df, aggcol = frames.insert_aggregated_in_hires(df=self.df.copy(), col=self.x_col,
                                                       to_freq='D', to_agg='max', agg_offset='7H')

        # Get hires TA and VPD data for nCHDs
        nchds_ix = (df[aggcol] >= thres_nchds_lower) & (df[aggcol] < thres_chds)
        nchds_hires_df = df.loc[nchds_ix, [self.ta_col, self.x_col]].copy()

        # Calculate diel cycles (half-hourly medians) for TA and VPD for nCHDs
        diel_cycles_nchds_df, ta_template_col, vpd_template_col = \
            self._diel_cycle(df=nchds_hires_df, agg='median')

        # Time as column for merging
        diel_cycles_nchds_df.index.name = 'INDEX'  # Rename to avoid same name as TIME column
        diel_cycles_nchds_df['TIME'] = diel_cycles_nchds_df.index
        df['TIME'] = df.index.time

        # Add diel cycles from nCHDs to full data, merge on time
        df['TIMESTAMP'] = df.index  # Original index
        df = df.merge(diel_cycles_nchds_df, left_on='TIME', right_on='TIME', how='left')
        df = df.set_index('TIMESTAMP')  # Re-apply original index (merging lost index)

        # Remove CHD data and replace with TEMPLATE (TA, VPD) or gap-filled (NEE)

        # Columns from which CHD data will be removed
        ta_limited_col = 'TA_LIMITED'  # CHD data will be replaced with nCHD diel cycle median
        vpd_limited_col = 'VPD_LIMITED'  # CHD data will be replaced with nCHD diel cycle median
        nee_limited_col = 'NEE_LIMITED'  # Will be gap-filled with random forest

        # Copy current available data
        df[ta_limited_col] = df[self.ta_col].copy()
        df[vpd_limited_col] = df[self.x_col].copy()
        df[nee_limited_col] = df[self.nee_col].copy()

        # Remove CHD data, creates gaps
        chds_ix = df[aggcol] >= thres_chds  # Indices of CHD data
        df.loc[chds_ix, ta_limited_col] = np.nan
        df.loc[chds_ix, vpd_limited_col] = np.nan
        df.loc[chds_ix, nee_limited_col] = np.nan

        # Fill gaps in TA and VPD with diel cycle nCHD medians
        df[ta_limited_col].fillna(df[ta_template_col], inplace=True)
        df[vpd_limited_col].fillna(df[vpd_template_col], inplace=True)

        # # Plot check
        # import matplotlib.pyplot as plt
        # df[[ta_template_col, ta_limited_col, ta_col]].plot(title=ta_col, xlim=('2019-06-15', '2019-07-15'))
        # df[[vpd_template_col, vpd_limited_col, x_col]].plot(title=x_col, xlim=('2019-06-15', '2019-07-15'))
        # df[[nee_col, nee_limited_col]].plot(title=nee_col, xlim=('2019-06-15', '2019-07-15'))
        # plt.show()

        return df, ta_limited_col, vpd_limited_col, nee_limited_col

    def _diel_cycle(self, df: DataFrame, agg: str or dict) -> tuple[DataFrame, str, str]:
        """Calculate diel cycles grouped by time"""
        diel_cycles_df = DataFrame(df)
        diel_cycles_df['TIME'] = diel_cycles_df.index.time
        diel_cycles_df = diel_cycles_df.groupby('TIME').agg(agg)
        new_ta_col = f"TA_TEMPLATE"
        # new_ta_col = f"{self.ta_col}_nchds_{agg}"
        new_vpd_col = f"VPD_TEMPLATE"
        frames.rename_cols(df=diel_cycles_df, renaming_dict={self.ta_col: new_ta_col,
                                                             self.x_col: new_vpd_col})
        # diel_cycles_df.columns = [f"{col}_nchds_{agg}" for col in diel_cycles_df.columns]  # Rename column
        # templates_df = diel_cycles_df[[(self.ta_col, 'q50'), (self.x_col, 'q50')]].copy()
        # diel_cycles_df.columns = ['_'.join(col).strip() for col in diel_cycles_df.columns.values]  # Make one-row header
        return diel_cycles_df, new_ta_col, new_vpd_col

    def _detect_criticalheatdays(self, usebins, bootstrap_runs):
        """Run analyses for critical heat days"""
        # Critical heat days
        chd = CriticalHeatDays(
            df=self.df,
            x_col=self.x_col,
            nee_col=self.nee_col,
            gpp_col=self.gpp_col,
            reco_col=self.reco_col,
            daynight_split='timestamp',
            usebins=usebins,
            # usebins=self.chd_usebins,
            bootstrap_runs=bootstrap_runs,
            # bootstrap_runs=self.chd_bootstrap_runs,
            bootstrapping_random_state=None
        )

        # Run CHD analyses
        chd.detect_chd_threshold()
        chd.analyze_daytime()
        chd.find_nee_optimum_range()

        # Provide CHD results to class
        self._results_chd_threshold_detection = chd.results_threshold_detection
        self._results_chd_flux_analysis = chd.results_daytime_analysis
        self._results_chd_optimum_range = chd.results_optimum_range

        return chd

    def plot_chd_threshold_detection(self, ax, highlight_year: int = None):
        self.chd_instance.plot_chd_detection_from_nee(ax=ax, highlight_year=highlight_year)

    def plot_daytime_analysis(self, ax, highlight_year: int = None):
        self.chd_instance.plot_daytime_analysis(ax=ax)

    def plot_rolling_bin_aggregates(self, ax):
        self.chd_instance.plot_rolling_bin_aggregates(ax=ax)

    def plot_bin_aggregates(self, ax):
        self.chd_instance.plot_bin_aggregates(ax=ax)

    def plot_vals_in_optimum_range(self, ax):
        self.chd_instance.plot_vals_in_optimum_range(ax=ax)

    def plot_cumulatives(self, ax):

        df = self.carboncost_df
        cumulative_orig = df['NEE_CUT_f'].cumsum()  # Cumulative of original measured and gap-filled NEE
        cumulative_model = df['NEE_LIMITED_gfRF'].cumsum()  # NEE where hot days were modeled

        # # Modeled data starting from where the cumulative curve deviates from orig
        # cumulative_model = cumulative_model.loc['2019-06-20 00:15:00':]

        # # Original data starting from where the modeled curve deviates from orig curve
        # cumulative_orig_modelrange = cumulative_orig.loc['2019-06-20 00:15:00':]

        # cumu_orig = cumulative_orig.iloc[-1]
        # cumu_model = cumulative_model.iloc[-1]
        # carbon_cost = cumu_model - cumu_orig
        # carbon_cost_rel = ((cumu_model / cumu_orig) - 1) * 100

        # # Fill between lines: net emission
        # net_emission = cumulative_orig.loc['2019-01-01 00:15:00':'2019-05-18 13:15:00']
        # ax.fill_between(net_emission.index, net_emission, 0, alpha=1, color='#ef9a9a',
        #                 zorder=1)  # uncertainty lines (95% confidence)
        # ax.text('2019-03-13', 10, "net emission", size=16, horizontalalignment='center', verticalalignment='bottom',
        #         color='white', weight='bold')

        # # Fill between lines: net uptake
        # net_uptake = cumulative_orig.loc['2019-05-18 13:15:00':]
        # ax.fill_between(net_uptake.index, net_uptake, 0, alpha=1, color='#AED581',
        #                 zorder=1)  # uncertainty lines (95% confidence)
        # ax.text('2019-09-15', -10, "net uptake", size=16, horizontalalignment='center', verticalalignment='top',
        #         color='white', weight='bold')

        # # Fill between: carbon cost
        # mpl.rcParams['hatch.linewidth'] = 2  # Set width of hatch lines
        # ax.fill_between(cumulative_model.index, cumulative_model, cumulative_orig_modelrange, alpha=1, lw=0,
        #                 color='#ef9a9a', edgecolor='white', zorder=1, hatch='//', label="carbon cost")

        # Original data as measured and gap-filled
        x = cumulative_orig.index
        y = cumulative_orig
        color = '#455A64'
        ax.plot_date(x=x, y=y, color=color, alpha=0.9, ls='-', lw=3, marker='', markeredgecolor='none', ms=0, zorder=99,
                     label='observed')
        ax.plot_date(x[-1], y[-1], ms=10, zorder=100, color=color)
        ax.text(x[-1], y[-1], f"    {cumulative_orig[-1]:.0f}", size=20, color=color, backgroundcolor='none', alpha=1,
                horizontalalignment='left', verticalalignment='center')

        # Modeled hot days
        x = cumulative_model.index
        y = cumulative_model
        color = '#b71c1c'
        ax.plot_date(x=x, y=y, color=color, alpha=0.9, ls='-', lw=3, marker='', markeredgecolor='none', ms=0, zorder=98,
                     label='without critical heat days')
        ax.plot_date(x[-1], y[-1], ms=10, zorder=100, color=color)
        ax.text(x[-1], y[-1], f"    {cumulative_model[-1]:.0f}", size=20, color=color, backgroundcolor='none', alpha=1,
                horizontalalignment='left', verticalalignment='center')

        # # Show days in original that reached max TA > 23°C
        # color='#F06292'
        # _df = subset_2019_df[['NEE', 'TA_DAILYMAX']].copy()
        # _df['NEE_CUMSUM'] = _df['NEE'].cumsum()
        # filter = (_df['TA_DAILYMAX'] > 23)
        # warm_days = _df.loc[filter]
        # ax.plot_date(x=warm_days.index, y=warm_days['NEE_CUMSUM'],
        #              alpha=1, ls='none', lw=0, ms=10, color=color,
        #              marker='.', markeredgecolor='none', zorder=99, label='Days where max TA > 23°C')

        ax.set_title("Carbon flux 2019", x=0.95, y=0.95, size=24, ha='right', weight='bold')
        ax.text(0.95, 0.93, "gC $\mathregular{m^{-2}}$", size=20, color='#9E9E9E', backgroundcolor='none', alpha=1,
                horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, weight='bold')

        ax.set_xlim(df.index[0], df.index[-1])
        # ax.set_ylim(-360, 100)

        ax.legend(ncol=1, edgecolor='none', loc=(.1, .1), prop={'size': 16})
        ax.axhline(0, lw=1, color='black')

        ax.tick_params(axis='both', which='major', direction='in', labelsize=16, length=8, size=5)  # x/y ticks text
        # xtickrange = range(16, 30, 2)
        # ax.set_xticks(['2019-01-01', '2019-03-01', '2019-05-01', '2019-07-01', '2019-09-01', '2019-11-01'])
        # ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])

        ax.set_xlabel('date', fontsize=16)
        ax.set_ylabel('cumulative carbon flux (gC $\mathregular{m^{-2}}$)', fontsize=16)

        # ax.legend(ncol=1, edgecolor='none', loc='lower right', prop={'size': 16})
        # ax.text(0.03, 0.96, 'Daily cumulative carbon flux',
        #             size=16, color='black', backgroundcolor='none', transform=ax.transAxes,
        #             alpha=0.8, horizontalalignment='left', verticalalignment='top')

        # Nice format for date ticks
        locator = mdates.AutoDateLocator(minticks=12, maxticks=20)
        ax.xaxis.set_major_locator(locator)
        formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
        ax.xaxis.set_major_formatter(formatter)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # fig.savefig('Figure2.png', bbox_inches='tight')

        # # Critical heat days
        # fig = plt.figure(figsize=(9, 9))
        # gs = gridspec.GridSpec(1, 1)  # rows, cols
        # # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
        # ax = fig.add_subplot(gs[0, 0])
        # chd.plot_chd_detection(ax=ax)
        # fig.show()

        # # Analyze flux
        # fig = plt.figure(figsize=(9, 9))
        # gs = gridspec.GridSpec(1, 1)  # rows, cols
        # # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
        # ax = fig.add_subplot(gs[0, 0])
        # chd.plot_flux(ax=ax, flux='nee', highlight_year=2019)
        # fig.show()

        # # Optimum range
        # fig = plt.figure(figsize=(9, 16))
        # gs = gridspec.GridSpec(3, 1)  # rows, cols
        # # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
        # ax1 = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[1, 0])
        # ax3 = fig.add_subplot(gs[2, 0])
        # chd.plot_rolling_bin_aggregates(ax=ax1)
        # chd.plot_bin_aggregates(ax=ax2)
        # chd.plot_vals_in_optimum_range(ax=ax3)
        # fig.show()


if __name__ == '__main__':
    from tests.testdata.loadtestdata import loadtestdata

    # Settings
    x_col = 'VPD_f'
    nee_col = 'NEE_CUT_f'
    gpp_col = 'GPP_DT_CUT'
    reco_col = 'Reco_DT_CUT'
    ta_col = 'Tair_f'
    swin_col = 'Rg_f'
    keepcols = [x_col, nee_col, gpp_col, reco_col, ta_col, swin_col]

    # Load data and make subset
    data_df, metadata_df = loadtestdata(
        filetype='CSV_TS-FULL-MIDDLE_30MIN',
        filepath=r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2016-2020.diive.csv')
        # filepath=r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2016-2020_mayToSep.csv')
    subset_data_df = data_df[keepcols].copy()

    # Carbon cost
    cc = CarbonCost(
        df=subset_data_df,
        x_col=x_col,
        nee_col=nee_col,
        gpp_col=gpp_col,
        reco_col=reco_col,
        ta_col=ta_col,
        swin_col=swin_col,
        focus_months_from=5,
        focus_months_to=9
    )

    # # Critical heat days
    # cc.detect_criticalheatdays(usebins=0, bootstrap_runs=10)
    #
    # ## Plots
    # fig = plt.figure(figsize=(9, 20))
    # gs = gridspec.GridSpec(5, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax3 = fig.add_subplot(gs[2, 0])
    # ax4 = fig.add_subplot(gs[3, 0])
    # ax5 = fig.add_subplot(gs[4, 0])
    # cc.plot_chd_threshold_detection(ax=ax1, highlight_year=2019)
    # cc.plot_daytime_analysis(ax=ax2, highlight_year=2019)
    # cc.plot_rolling_bin_aggregates(ax=ax3)
    # cc.plot_bin_aggregates(ax=ax4)
    # cc.plot_vals_in_optimum_range(ax=ax5)
    # fig.show()

    # Gapfill critical heat days
    cc.gapfill_criticalheatdays()

    ## Plot
    fig = plt.figure(facecolor='white', figsize=(9, 9))
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.2, hspace=0.25, left=0.06, right=0.97, top=0.97, bottom=0.03)

    ax = fig.add_subplot(gs[0, 0])
    cc.plot_cumulatives(ax=ax)
    fig.show()

    # # # Insert aggregated values in high-res dataframe
    # # _df, agg_col = insert_aggregated_in_hires(df=df, col=x_col, to_freq='D', to_agg='max')
    # #
    # # thres_nchds_upper = cc.results_chd_threshold_detection['thres_nchds_upper']
    # # thres_nchds_lower = cc.results_chd_threshold_detection['thres_nchds_lower']
    # # # thres_nchds_upper = chd.results_chd['thres_nchds_upper']
    # # # thres_nchds_lower = chd.results_chd['thres_nchds_lower']
    # #
    # # # Get nCHDs
    # # filter_nchds = (_df[agg_col] >= thres_nchds_lower) & (_df[agg_col] <= thres_nchds_upper)
    # # nchds_df = _df.loc[filter_nchds, :]
    # #
    # # # Build template diel cycle
    # # # Build template from nCHDs
    # # diel_cycle_df = nchds_df.copy()
    # # diel_cycle_df['TIME'] = diel_cycle_df.index.time
    # # aggs = {'mean', 'min', 'max', 'median'}
    # # diel_cycle_df = diel_cycle_df.groupby('TIME').agg(aggs)
    # #
    # # diel_cycle_df[ta_col][['mean', 'max', 'min', 'median']].plot(title="TA 1997-2019 for nCHDs")
    # # plt.show()
    # #
    # # # # todo SetToMissingVals: set values to missing based on condition
    # # #
    # # # # todo fill gaps with diel cycle from specified time period
    #
    # print("END")
