import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas import DataFrame

from diive.common.dfun.frames import insert_aggregated_in_hires
from diive.pkgs.flux.criticalheatdays import CriticalHeatDays


class CarbonCost:

    def __init__(
            self,
            df: DataFrame,
            x_col: tuple,
            nee_col: tuple,
            gpp_col: tuple,
            reco_col: tuple,
            daytime_col: tuple,
            daytime_threshold: int = 20,
            set_daytime_if: str = 'Larger Than Threshold',
            chd_usebins: int = 10,
            chd_bootstrap_runs: int = 10
    ):
        self.df = df
        self.x_col = x_col
        self.nee_col = nee_col
        self.gpp_col = gpp_col
        self.reco_col = reco_col
        self.daytime_col = daytime_col
        self.daytime_threshold = daytime_threshold
        self.set_daytime_if = set_daytime_if
        self.chd_usebins = chd_usebins
        self.chd_bootstrap_runs = chd_bootstrap_runs

        # Results from critical heat days analyses
        self._results_chd_threshold_detection = None
        self._results_chd_flux_analysis = None
        self._results_chd_optimum_range = None
        self._chd_instance = None

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
        """Return results for optimum range"""
        if not self._chd_instance:
            raise Exception('No instance for CriticalHeatDays found')
        return self._chd_instance

    def run(self):
        self._chd_instance = self._criticalheatdays()

    def _criticalheatdays(self):
        """Run analyses for critical heat days"""
        # Critical heat days (CHDs)
        chd = CriticalHeatDays(
            df=self.df,
            x_col=self.x_col,
            flux_col=self.nee_col,
            gpp_col=self.gpp_col,
            reco_col=self.reco_col,
            daynight_split=self.daytime_col,
            daytime_threshold=self.daytime_threshold,
            set_daytime_if=self.set_daytime_if,
            usebins=self.chd_usebins,
            bootstrap_runs=self.chd_bootstrap_runs
        )

        # Run CHD analyses
        chd.detect_chd_threshold_from_partitioned()
        chd.analyze_flux(flux='nee')
        chd.find_nee_optimum_range()

        # Provide CHD results to class
        self._results_chd_threshold_detection = chd.results_threshold_detection
        self._results_chd_flux_analysis = chd.results_daytime_analysis
        self._results_chd_optimum_range = chd.results_optimum_range

        return chd

    def plot_chd_threshold_detection(self, ax):
        self.chd_instance.plot_chd_detection_from_nee(ax=ax)

    def plot_chd_flux_analysis(self, ax):
        self.chd_instance.plot_daytime_analysis(ax=ax, flux='nee', highlight_year=2019)

    def plot_rolling_bin_aggregates(self, ax):
        self.chd_instance.plot_rolling_bin_aggregates(ax=ax)

    def plot_bin_aggregates(self, ax):
        self.chd_instance.plot_bin_aggregates(ax=ax)

    def plot_vals_in_optimum_range(self, ax):
        self.chd_instance.plot_vals_in_optimum_range(ax=ax)

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

    df = loadtestdata()

    # Use data from May to Sep only
    maysep_filter = (df.index.month >= 5) & (df.index.month <= 9)
    df = df.loc[maysep_filter].copy()

    # Settings
    x_col = ('VPD_f', '-no-units-')
    nee_col = ('NEE_CUT_f', '-no-units-')
    gpp_col = ('GPP_DT_CUT', '-no-units-')
    reco_col = ('Reco_DT_CUT', '-no-units-')
    daytime_col = ('Rg_f', '-no-units-')
    ta_col = ('Tair_f', '-no-units-')

    cc = CarbonCost(
        df=df,
        x_col=x_col,
        nee_col=nee_col,
        gpp_col=gpp_col,
        reco_col=reco_col,
        daytime_col=daytime_col,
        daytime_threshold=50,
        set_daytime_if='Larger Than Threshold',
        chd_usebins=0,
        chd_bootstrap_runs=5
    )

    cc.run()

    # Plots
    fig = plt.figure(figsize=(9, 16))
    gs = gridspec.GridSpec(2, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # ax3 = fig.add_subplot(gs[2, 0])
    # ax4 = fig.add_subplot(gs[3, 0])
    # ax5 = fig.add_subplot(gs[4, 0])
    cc.plot_chd_threshold_detection(ax=ax1)
    cc.plot_chd_flux_analysis(ax=ax2)
    # cc.plot_rolling_bin_aggregates(ax=ax3)
    # cc.plot_bin_aggregates(ax=ax4)
    # cc.plot_vals_in_optimum_range(ax=ax5)
    fig.show()




    # # Insert aggregated values in high-res dataframe
    # _df, agg_col = insert_aggregated_in_hires(df=df, col=x_col, to_freq='D', to_agg='max')
    #
    # thres_nchds_upper = cc.results_chd_threshold_detection['thres_nchds_upper']
    # thres_nchds_lower = cc.results_chd_threshold_detection['thres_nchds_lower']
    # # thres_nchds_upper = chd.results_chd['thres_nchds_upper']
    # # thres_nchds_lower = chd.results_chd['thres_nchds_lower']
    #
    # # Get nCHDs
    # filter_nchds = (_df[agg_col] >= thres_nchds_lower) & (_df[agg_col] <= thres_nchds_upper)
    # nchds_df = _df.loc[filter_nchds, :]
    #
    # # Build template diel cycle
    # # Build template from nCHDs
    # diel_cycle_df = nchds_df.copy()
    # diel_cycle_df['TIME'] = diel_cycle_df.index.time
    # aggs = {'mean', 'min', 'max', 'median'}
    # diel_cycle_df = diel_cycle_df.groupby('TIME').agg(aggs)
    #
    # diel_cycle_df[ta_col][['mean', 'max', 'min', 'median']].plot(title="TA 1997-2019 for nCHDs")
    # plt.show()
    #
    # # # todo SetToMissingVals: set values to missing based on condition
    # #
    # # # todo fill gaps with diel cycle from specified time period

    print("END")
