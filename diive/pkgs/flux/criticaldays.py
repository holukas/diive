"""
=============
CRITICAL DAYS
=============
"""
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting import plotfuncs
from diive.core.plotting.plotfuncs import save_fig
from diive.core.plotting.rectangle import rectangle
from diive.core.plotting.styles.LightTheme import COLOR_THRESHOLD, \
    FONTSIZE_LEGEND, COLOR_NEE, INFOTXT_FONTSIZE
from diive.pkgs.fits._binfitter import BinFitterBTS, PlotBinFitterBTS

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


class CriticalDays:

    def __init__(
            self,
            df: DataFrame,
            x_col: str,
            y_col: str,
            thres_min_x: float or int,
            x_agg: Literal['max'] = 'max',
            y_agg: Literal['sum'] = 'sum',
            usebins: int = 10,
            n_bootstrap_runs: int = 10,
            bootstrapping_random_state: int = None,
            thres_from_bootstrap: Literal['max', 'median'] = 'max',
            thres_y_sign_change: Literal['+', '-'] = '-'
    ):
        """Detect threshold in x that defines critical days in y

        For example, investigate daily NEP sums (y) in relation to
        daily max VPD (x).

        Args:
            df: Data in half-hourly time resolution
            x_col: Column name of x variable, e.g. 'VPD'
            y_col: Column name of y variable, e.g. 'NEP'
            thres_min_x: Minimum value for x at threshold
                e.g., VPD must be min 1 kPa for threshold to be accepted
            x_agg: Daily aggregation for x, e.g. 'max' selects the daily maximum
            y_agg: Daily aggregation for y, e.g. 'sum' selects daily sums
            usebins: XXX (default 10)
            n_bootstrap_runs: Number of bootstrap runs during detection of
                critical heat days. Must be an odd number. In case an even
                number is given +1 is added automatically.
            bootstrapping_random_state: Option to use fixed random state
            thres_from_bootstrap: Which aggregate to use as threshold,
                e.g., for bootstrapping results [2.1, 2.2, 2.1, 2.5, 2.3] 'max'
                selects 2.5 as threshold
            thres_y_sign_change: Defines what the detected threshold describes, can
                be '+' or '-'
                e.g., if '+', the threshold describes the *x* value where
                    *y* changes from a negative sign to a positive sign
                e.g., if '-', the threshold describes the *x* value where
                    *y* changes from a positive sign to a negative sign
                e.g., for flux NEE if '+', the threshold describes the VPD
                    value where NEE becomes positive (i.e., net loss of CO2)
                e.g., for flux NEP if '-', the threshold describes the VPD
                    value where NEP becomes negative (i.e., net loss of CO2)
        """

        # self.df = df[[x_col, y_col]].copy()
        self.df = df.copy()
        self.x_col = x_col
        self.x_agg = x_agg
        self.y_col = y_col
        self.y_agg = y_agg
        self.usebins = usebins
        self.bootstrapping_random_state = bootstrapping_random_state
        self.thres_from_bootstrap = thres_from_bootstrap
        self.thres_y_sign_change = thres_y_sign_change
        self.thres_min_x = thres_min_x

        self.n_bootstrap_runs = n_bootstrap_runs

        # Resample dataset
        aggs = ['mean', 'median', 'count', 'min', 'max', 'sum']
        self.df_aggs = self._resample_dataset(df=self.df, aggs=aggs, day_start_hour=None)

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

    def detect_dcrit_threshold(self):
        """Detect critical days x threshold for y"""

        # Fit to bootstrapped data, daily time resolution
        fit_results, bts_fit_results = self._bootstrap_fits(df_aggs=self.df_aggs,
                                                            x_col=self.x_col,
                                                            x_agg=self.x_agg,
                                                            y_col=self.y_col,
                                                            y_agg=self.y_agg,
                                                            fit_to_bins=self.usebins)

        # Add info about zero crossings to fit results
        fit_results = self._collect_zerocrossings(fit_results=fit_results)
        for key, cur_fit_results in bts_fit_results.items():
            bts_fit_results[key] = self._collect_zerocrossings(fit_results=bts_fit_results[key])

        # Get flux zerocrossing values (y has crossed zeroline) from bootstrap runs
        bts_zerocrossings_df = self._bts_zerocrossings_collect(bts_fit_results=bts_fit_results)

        # Calc flux equilibrium points aggregates from bootstrap runs
        bts_zerocrossings_aggs = self._zerocrossings_aggs(bts_zerocrossings_df=bts_zerocrossings_df)

        # Threshold for Critical Heat Days (Dcrit)
        # defined as the linecrossing e.g. MAX x (e.g. VPD) from bootstrap runs
        # thres_dcrit = bts_zerocrossings_aggs['x_median']
        thres_dcrit = bts_zerocrossings_aggs[f"x_{self.thres_from_bootstrap}"]  # e.g. "x_max"

        # Collect days above or equal to Dcrit threshold
        df_aggs_dcrit = self.df_aggs.loc[self.df_aggs[self.x_col][self.x_agg] >= thres_dcrit, :].copy()

        # Number of days above Dcrit threshold
        n_dcrit = len(df_aggs_dcrit)

        # Collect Near-Critical Heat Days (nDcrit)
        # With the number of Dcrit known, collect data for the same number
        # of days below of equal to Dcrit threshold.
        # For example: if 10 Dcrit were found, nDcrit are the 10 days closest
        # to the Dcrit threshold (below or equal to the threshold).
        sortby_col = (self.x_col, self.x_agg)
        ndcrit_start_ix = n_dcrit
        ndcrit_end_ix = n_dcrit * 2
        df_aggs_ndcrit = self.df_aggs \
                             .sort_values(by=sortby_col, ascending=False) \
                             .iloc[ndcrit_start_ix:ndcrit_end_ix]

        # Threshold for nDcrit
        # The lower threshold is e.g. the minimum of found x maxima, depending
        # on the param "thres_from_bootstrap"
        thres_ndcrit_lower = df_aggs_ndcrit[self.x_col][self.x_agg].min()
        thres_ndcrit_upper = thres_dcrit

        # Number of days above nDcrit threshold and below or equal Dcrit threshold
        n_ndcrit = len(df_aggs_ndcrit)

        # Collect results
        self._results_threshold_detection = dict(
            fit_results=fit_results,  # Non-boostrapped fit results
            bts_zerocrossings_df=bts_zerocrossings_df,
            bts_zerocrossings_aggs=bts_zerocrossings_aggs,
            thres_dcrit=thres_dcrit,
            thres_ndcrit_lower=thres_ndcrit_lower,
            thres_ndcrit_upper=thres_ndcrit_upper,
            df_aggs_dcrit=df_aggs_dcrit,
            df_aggs_ndcrit=df_aggs_ndcrit,
            n_dcrit=n_dcrit,
            n_ndcrit=n_ndcrit,
        )

    def plot_dcrit_detection_results(self, ax,
                                     x_units: str,
                                     y_units: str,
                                     x_label: str = None,
                                     y_label: str = None,
                                     showfit: bool = True,
                                     highlight_year: int = None,
                                     showline_dcrit: bool = True,
                                     showrange_dcrit: bool = True,
                                     label_threshold: str = None):
        """Plot results from critical days threshold detection"""

        label_threshold = 'threshold' if not label_threshold else label_threshold

        # bts_results = self.results_threshold_detection['bts_results'][0]  # 0 means non-bootstrapped data
        fit_results = self.results_threshold_detection['fit_results']
        bts_zerocrossings_aggs = self.results_threshold_detection['bts_zerocrossings_aggs']

        self.ax, line_xy, line_highlight, line_fit, line_fit_ci, line_fit_pb = \
            PlotBinFitterBTS(fit_results=fit_results,
                             ax=ax, label="NEP",
                             highlight_year=highlight_year, edgecolor='#B0BEC5',
                             color='none', color_fitline=COLOR_NEE,
                             showfit=showfit, show_prediction_interval=True).plot_binfitter()

        # Threshold line
        # Vertical line showing e.g. MAX Dcrit threshold from bootstrap runs
        line_dcrit_vertical = None
        if showline_dcrit:
            thres_dcrit = self.results_threshold_detection['thres_dcrit']
            line_dcrit_vertical = ax.axvline(thres_dcrit, lw=3,
                                             color=COLOR_THRESHOLD, ls='-', zorder=99,
                                             label=f"{label_threshold} = {thres_dcrit:.2f} {x_units}")
            # Text Dcrit
            n_dcrit = len(self.results_threshold_detection['df_aggs_dcrit'])
            sym_max = r'$\rightarrow$'
            _pos = thres_dcrit + thres_dcrit * 0.02
            t = ax.text(_pos, 0.1, f"{sym_max} {n_dcrit} critical days {sym_max}", size=INFOTXT_FONTSIZE,
                        color='black', backgroundcolor='none', transform=ax.get_xaxis_transform(),
                        alpha=1, horizontalalignment='left', verticalalignment='top', zorder=99)
            t.set_bbox(dict(facecolor=COLOR_THRESHOLD, alpha=.7, edgecolor=COLOR_THRESHOLD))

            # # Area Dcrit
            # area_dcrit = ax.fill_between([bts_zerocrossings_aggs[_thres_agg],
            #                             fit_results['fit_df']['fit_x'].max()],
            #                            0, 1, color='#BA68C8', alpha=0.1,
            #                            transform=ax.get_xaxis_transform(),
            #                            label=f"Dcrit ({n_dcrit} days)", zorder=1)

        # ## Rectangle bootstrap range Dcrit
        range_bts_netzeroflux = None
        if showrange_dcrit:
            range_bts_netzeroflux = rectangle(
                ax=ax,
                rect_lower_left_x=bts_zerocrossings_aggs['x_min'],
                rect_lower_left_y=0,
                rect_width=bts_zerocrossings_aggs['x_max'] - bts_zerocrossings_aggs['x_min'],
                rect_height=1,
                label=f"{label_threshold} range ({self.n_bootstrap_runs} bootstraps)",
                color=COLOR_THRESHOLD)

        # Format
        ax.axhline(0, lw=1, color='black')
        if x_label:
            x_label = f"{x_label} ({x_units})"
        else:
            x_label = f"Daily {self.x_agg} {self.x_col} ({x_units})"
        if y_label:
            y_label = f"{y_label} ({y_units})"
        else:
            y_label = f"Daily {self.y_agg} {self.y_col} ({y_units})"
        plotfuncs.default_format(ax=ax, ax_xlabel_txt=x_label, ax_ylabel_txt=y_label)
        
        # Custom legend
        # This legend replaces the legend from PlotBinFitterBTS
        # Assign two of the handles to the same legend entry by putting them in a tuple
        # and using a generic handler map (which would be used for any additional
        # tuples of handles like (p1, p3)).
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
        l = ax.legend(
            [
                line_dcrit_vertical if showline_dcrit else None,
                range_bts_netzeroflux if showrange_dcrit else None,
                line_xy,
                line_highlight if line_highlight else None,
                line_fit if showfit else None,
                line_fit_ci if showfit else None,
                line_fit_pb if showfit else None
            ],
            [
                line_dcrit_vertical.get_label() if showline_dcrit else None,
                range_bts_netzeroflux.get_label() if showrange_dcrit else None,
                line_xy.get_label(),
                line_highlight.get_label() if line_highlight else None,
                line_fit.get_label() if showfit else None,
                line_fit_ci.get_label() if showfit else None,
                line_fit_pb.get_label() if showfit else None
            ],
            bbox_to_anchor=(0, 1),
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
            loc="lower left",
            ncol=2,
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)})

        ax.text(0.06, 0.95, "(a)",
                size=theme.AX_LABELS_FONTSIZE, color='black', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='left', verticalalignment='top')

        return ax

    def _resample_dataset(self, df: DataFrame, aggs: list, day_start_hour: int = None):
        """Resample to daily values from *day_start_hour* to *day_start_hour*"""
        if day_start_hour:
            df_aggs = df.resample('D', offset=f'{day_start_hour}H').agg(aggs)
        else:
            df_aggs = df.resample('D').agg(aggs)
        df_aggs = df_aggs.where(df_aggs[self.x_col]['count'] == 48)  # Full days only
        # df_aggs = df_aggs.where(df_aggs[self.x_col]['count'] == 48).dropna()  # Full days only
        df_aggs.index.name = 'TIMESTAMP_START'
        return df_aggs

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
            y_median=bts_zerocrossings_df['y_nom'].median(),
            y_min=bts_zerocrossings_df['y_nom'].min(),
            y_max=bts_zerocrossings_df['y_nom'].max(),
        )

        return zerocrossings_aggs

    def _bts_zerocrossings_collect(self, bts_fit_results) -> DataFrame:
        """Collect all zero crossing values in DataFrame"""
        bts_linecrossings_df = pd.DataFrame()
        for bts in range(0, self.n_bootstrap_runs):
            _dict = bts_fit_results[bts]['zerocrossing_vals']
            _series = pd.Series(_dict)
            _series.name = bts
            if bts == 0:
                bts_linecrossings_df = pd.DataFrame(_series).T
            else:
                bts_linecrossings_df = bts_linecrossings_df.append(_series.T)
        return bts_linecrossings_df

    def _bootstrap_fits(self,
                        df_aggs: DataFrame,
                        x_col: str,
                        x_agg: str,
                        y_col: str,
                        y_agg: str,
                        fit_to_bins: int = 10) -> tuple[dict, dict]:
        """Bootstrap ycols and fit to x"""

        # Get column names in aggregated df
        x_col = (x_col, x_agg)
        y_col = (y_col, y_agg)

        fitter = BinFitterBTS(
            df=df_aggs,
            n_bootstraps=self.n_bootstrap_runs,
            x_col=x_col,
            y_col=y_col,
            predict_max_x=self.predict_max_x,
            predict_min_x=self.predict_min_x,
            n_predictions=1000,
            n_bins_x=0,
            # bins_y_agg='mean',
            fit_type='quadratic'
        )
        fitter.fit()

        fit_results = fitter.fit_results
        bts_fit_results = fitter.bts_fit_results

        return fit_results, bts_fit_results

    def _collect_zerocrossings(self, fit_results) -> dict:
        """Collect zero-crossing results from all fit results"""
        zerocrossing_vals = self._detect_zerocrossing_y(fit_df=fit_results['fit_df'])
        if isinstance(zerocrossing_vals, dict):
            pass
        else:
            raise ValueError
        fit_results['zerocrossing_vals'] = zerocrossing_vals
        print(f"Found zero-crossing values: {fit_results['zerocrossing_vals']}")
        return fit_results

    def _detect_zerocrossing_y(self, fit_df: DataFrame):
        # kudos: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value

        n_zerocrossings = None
        zerocrossings_ix = None

        # Collect predicted vals in df
        zerocrossings_df = pd.DataFrame()
        zerocrossings_df['x_col'] = fit_df['fit_x']
        zerocrossings_df['y_nom'] = fit_df['nom']

        # Check values above/below zero
        _signs = np.sign(zerocrossings_df['y_nom'])
        _signs_max = _signs.max()
        _signs_min = _signs.min()

        if _signs_max == _signs_min:
            print("y does not cross zero-line.")
        else:
            zerocrossings_ix = np.argwhere(np.diff(_signs)).flatten()
            n_zerocrossings = len(zerocrossings_ix)

        # There must be one single line crossing to accept result
        # If there is more than one line crossing, reject result
        if n_zerocrossings == 1:
            zerocrossings_ix = zerocrossings_ix[0]
        else:
            raise ("More than 1 zero-crossing detected. "
                   "There must be one single line crossing to accept result. "
                   "Stopping.")

        # Values at zero crossing needed
        # Found index is last element *before* zero-crossing, therefore + 1
        zerocrossing_vals = zerocrossings_df.iloc[zerocrossings_ix + 1]
        zerocrossing_vals = zerocrossing_vals.to_dict()

        # Value for y at zero crossing
        # i.e. reject if NEE after zero-crossing does not change to emission

        # Check if the sign after the crossing is indeed negative as expected
        if (self.thres_y_sign_change == '-') & (zerocrossing_vals['y_nom'] < 0):
            pass
        # Check if the sign after the crossing is indeed positive as expected
        elif (self.thres_y_sign_change == '+') & (zerocrossing_vals['y_nom'] > 0):
            pass
        # If the sign after the crossing is positive against expectations, return None
        elif (self.thres_y_sign_change == '-') & (zerocrossing_vals['y_nom'] > 0):
            return None
        # If the sign after the crossing is negative against expectations, return None
        elif (self.thres_y_sign_change == '+') & (zerocrossing_vals['y_nom'] < 0):
            return None

        # x value must be above threshold to be somewhat meaningful, otherwise reject result
        if (zerocrossing_vals['x_col'] < self.thres_min_x):
            # x is too low, must be at least 1 kPa for valid crossing
            return None

        return zerocrossing_vals

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

    def showplot_criticaldays(self,
                              saveplot: bool = False,
                              title: str = None,
                              path: Path or str = None,
                              dpi: int = 72,
                              **kwargs):
        fig = plt.figure(figsize=(9, 9), dpi=dpi)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self.plot_dcrit_detection_results(ax=ax, **kwargs)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)
        return ax


if __name__ == '__main__':
    # # Subset
    # subset_cols = [vpd_col, nee_col]
    # df = df_orig[subset_cols].copy().dropna()
    #
    # # Convert units
    # df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    # df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    # df = df.loc[df.index.year <= 2022]
    # df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)]

    from diive.core.io.files import load_pickle

    # Variables
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    x_col = vpd_col
    y_col = nee_col

    # Load data, using pickle for fast loading
    source_file = r"L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\__apply\co2penalty_dav\input_data\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle"
    df_orig = load_pickle(filepath=source_file)
    df_orig = df_orig.loc[df_orig.index.year <= 2022].copy()
    # df_orig = df_orig.loc[df_orig.index.year >= 2019].copy()

    # Select daytime data between May and September 1997-2021
    maysep_dt_df = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()

    # Convert units
    maysep_dt_df[vpd_col] = maysep_dt_df[vpd_col].multiply(0.1)  # hPa --> kPa
    maysep_dt_df[nee_col] = maysep_dt_df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    x_units = "kPa"
    y_units = "gCO_{2}\ m^{-2}\ 30min^{-1}"
    xlabel = f"Half-hourly VPD ({x_units})"
    ylabel = f"{y_col} (${y_units}$)"

    # Critical days
    dcrit = CriticalDays(
        df=maysep_dt_df,
        x_col=vpd_col,
        x_agg='max',
        y_col=nee_col,
        y_agg='sum',
        usebins=0,
        n_bootstrap_runs=99,
        bootstrapping_random_state=None,
        thres_from_bootstrap='max'
    )
    dcrit.detect_dcrit_threshold()
    results_threshold_detection = dcrit.results_threshold_detection
    dcrit.showplot_criticaldays(x_units='kPa',
                                y_units="gCO_{2}\ m^{-2}\ d^{-1}",
                                highlight_year=2019,
                                showline_dcrit=True,
                                showrange_dcrit=True)
