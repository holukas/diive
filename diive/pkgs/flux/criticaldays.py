"""
FLUX: CRITICAL HEAT DAYS
========================
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
from diive.core.dfun.fits import BinFitterCP
from diive.core.plotting import plotfuncs
from diive.core.plotting.fitplot import fitplot
from diive.core.plotting.plotfuncs import save_fig
from diive.core.plotting.rectangle import rectangle
from diive.core.plotting.styles.LightTheme import COLOR_THRESHOLD, \
    FONTSIZE_LEGEND, COLOR_THRESHOLD2, COLOR_NEE, INFOTXT_FONTSIZE

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


class CriticalDays:

    def __init__(
            self,
            df: DataFrame,
            x_col: str,
            x_units: str,
            y_col: str,
            y_units: str,
            x_agg: Literal['max'] = 'max',
            y_agg: Literal['sum'] = 'sum',
            usebins: int = 10,
            bootstrap_runs: int = 10,
            bootstrapping_random_state: int = None
    ):
        """Detect critical heat days from ecosystem fluxes

        XXX

        Args:
            df:
            x_col: Column name of x variable
            y_col: Column name of NEE (net ecosystem exchange, ecosystem flux)
            usebins: XXX (default 10)
            bootstrap_runs: Number of bootstrap runs during detection of
                critical heat days. Must be an odd number. In case an even
                number is given +1 is added automatically.
        """

        self.df = df[[x_col, y_col]].copy()
        self.x_col = x_col
        self.x_units = x_units
        self.x_agg = x_agg
        self.y_col = y_col
        self.y_units = y_units
        self.y_agg = y_agg
        self.usebins = usebins
        self.bootstrapping_random_state = bootstrapping_random_state

        # Number of bootstrap runs must be odd number
        if bootstrap_runs % 2 == 0:
            bootstrap_runs += 1
        self.bootstrap_runs = bootstrap_runs

        # Resample dataset
        aggs = ['mean', 'median', 'count', 'min', 'max', 'sum']
        self.df_aggs = self._resample_dataset(df=self.df, aggs=aggs, day_start_hour=7)

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

    def _detect_xcrd_threshold(self):
        """Detect extremely critical x threshold for y"""
        _x = (self.x_col, self.x_agg)
        _y = (self.y_col, self.y_agg)
        xcrd_subset = self.df_aggs[[_x, _y]].copy()
        xcrd_subset = xcrd_subset.sort_values(by=_x, ascending=False)
        xcrd_subset.reset_index(inplace=True)  # Also inserts index as col
        xcrd_subset_ybelowzero = xcrd_subset.loc[xcrd_subset[_y] < 0, :].copy()  # todo check < 0, make setting for this
        thres_xcrd = xcrd_subset_ybelowzero[_x].max()
        num_xcrds = len(xcrd_subset.loc[xcrd_subset[_x] > thres_xcrd])
        print(f"Extreme threshold: {thres_xcrd}")
        return thres_xcrd, num_xcrds

    def detect_crd_threshold(self):
        """Detect critical days x threshold for y"""

        thres_xcrd, num_xcrds = self._detect_xcrd_threshold()

        # Fit to bootstrapped data, daily time resolution
        # Stored as bootstrap runs > 0 (bts>0)
        bts_results = self._bootstrap_fits(df_aggs=self.df_aggs,
                                           x_col=self.x_col,
                                           x_agg=self.x_agg,
                                           y_col=self.y_col,
                                           y_agg=self.y_agg,
                                           fit_to_bins=self.usebins,
                                           detect_zerocrossing=True)

        # Get flux zerocrossing values (y has crossed zeroline) from bootstrap runs
        bts_zerocrossings_df = self._bts_zerocrossings_collect(bts_results=bts_results)

        # Calc flux equilibrium points aggregates from bootstrap runs
        bts_zerocrossings_aggs = self._zerocrossings_aggs(bts_zerocrossings_df=bts_zerocrossings_df)

        # Threshold for Critical Heat Days (CRDs)
        # defined as the linecrossing MEDIAN x (e.g. VPD) from bootstrap runs
        thres_crd = bts_zerocrossings_aggs['x_median']
        # thres_crd = bts_zerocrossings_aggs['x_max']

        # Collect days above or equal to CRD threshold
        df_aggs_crds = self.df_aggs.loc[self.df_aggs[self.x_col]['max'] >= thres_crd, :].copy()

        # Number of days above CRD threshold
        num_crds = len(df_aggs_crds)

        # Collect Near-Critical Heat Days (nCRDs)
        # With the number of CRDs known, collect data for the same number
        # of days below of equal to CRD threshold.
        # For example: if 10 CRDs were found, nCRDs are the 10 days closest
        # to the CRD threshold (below or equal to the threshold).
        sortby_col = (self.x_col, 'max')
        ncrds_start_ix = num_crds
        ncrds_end_ix = num_crds * 2
        df_aggs_ncrds = self.df_aggs \
                            .sort_values(by=sortby_col, ascending=False) \
                            .iloc[ncrds_start_ix:ncrds_end_ix]

        # Threshold for nCRDs
        # The lower threshold is the minimum of found x maxima
        thres_ncrds_lower = df_aggs_ncrds[self.x_col]['max'].min()
        thres_ncrds_upper = thres_crd

        # Number of days above nCRD threshold and below or equal CRD threshold
        num_ncrds = len(df_aggs_ncrds)

        # Collect results
        self._results_threshold_detection = dict(
            bts_results=bts_results,
            bts_zerocrossings_df=bts_zerocrossings_df,
            bts_zerocrossings_aggs=bts_zerocrossings_aggs,
            thres_crd=thres_crd,
            thres_xcrd=thres_xcrd,
            thres_ncrds_lower=thres_ncrds_lower,
            thres_ncrds_upper=thres_ncrds_upper,
            df_aggs_crds=df_aggs_crds,
            df_aggs_ncrds=df_aggs_ncrds,
            num_crds=num_crds,
            num_ncrds=num_ncrds,
            num_xcrds=num_xcrds
        )

    def plot_crd_detection_results(self, ax,
                                   showfit: bool = True,
                                   highlight_year: int = None,
                                   showline_crd: bool = True,
                                   showrange_crd: bool = True,
                                   showline_xcrd: bool = True):
        """Plot results from critical heat days threshold detection"""

        # reload(diive.core.plotting.styles.LightTheme)
        # import diive.core.plotting.styles.LightTheme

        bts_results = self.results_threshold_detection['bts_results'][0]  # 0 means non-bootstrapped data
        bts_zerocrossings_aggs = self.results_threshold_detection['bts_zerocrossings_aggs']

        # y
        line_xy_y = line_fit_y = line_fit_ci_y = line_fit_pb_y = line_highlight = None
        _numvals_per_bin = bts_results[self.y_col]['numvals_per_bin']
        flux_bts_results = bts_results[self.y_col]
        line_xy_y, line_fit_y, line_fit_ci_y, line_fit_pb_y, line_highlight = \
            fitplot(ax=ax,
                    label="NEE",
                    highlight_year=highlight_year,
                    flux_bts_results=flux_bts_results,
                    edgecolor='#B0BEC5', color='none', color_fitline=COLOR_NEE,
                    showfit=showfit, show_prediction_interval=True)

        # # Actual non-bootstrapped line crossing, the point where RECO = GPP
        # line_netzeroflux = ax.scatter(bts_results['zerocrossing_vals']['x_col'],
        #                               bts_results['zerocrossing_vals']['y_nom'],
        #                               edgecolor='black', color='none', alpha=1, s=90, lw=2,
        #                               label='net zero flux', zorder=99, marker='s')

        # # Rectangle
        # # Bootstrapped line crossing, the point where RECO = GPP
        # line_bts_median_netzeroflux = ax.scatter(bts_zerocrossings_aggs['x_max'],
        #                                          0,
        #                                          edgecolor=COLOR_NEE, color='none', alpha=1, s=350, lw=3,
        #                                          label='net zero flux (bootstrapped median)', zorder=97, marker='s')

        # Threshold lines

        ## Vertical line showing MEDIAN CRD threshold from bootstrap runs
        line_crd_vertical = None
        if showline_crd:
            _sub = "$_{CRD}$"
            line_crd_vertical = ax.axvline(bts_zerocrossings_aggs['x_median'], lw=3, color=COLOR_THRESHOLD, ls='-',
                                           zorder=99,
                                           label=f"THR{_sub}, VPD = {bts_zerocrossings_aggs['x_median']:.2f} {self.x_units}")
            # # CRDs
            num_crds = len(self.results_threshold_detection['df_aggs_crds'])
            # area_crd = ax.fill_between([bts_zerocrossings_aggs['x_max'],
            #                             flux_bts_results['fit_df']['fit_x'].max()],
            #                            0, 1,
            #                            color='#BA68C8', alpha=0.1, transform=ax.get_xaxis_transform(),
            #                            label=f"CRDs ({num_crds} days)", zorder=1)
            sym_max = r'$\rightarrow$'
            _pos = bts_zerocrossings_aggs['x_median'] + bts_zerocrossings_aggs['x_median'] * 0.02
            t = ax.text(_pos, 0.1, f"{sym_max} {num_crds} critical days {sym_max}", size=INFOTXT_FONTSIZE,
                        color='black', backgroundcolor='none', transform=ax.get_xaxis_transform(),
                        alpha=1, horizontalalignment='left', verticalalignment='top', zorder=99)
            t.set_bbox(dict(facecolor=COLOR_THRESHOLD, alpha=.7, edgecolor=COLOR_THRESHOLD))

        ## Rectangle bootstrap range CRD
        range_bts_netzeroflux = None
        if showrange_crd:
            num_bootstrap_runs = len(
                self.results_threshold_detection['bts_results'].keys()) - 1  # -1 b/c zero run is non-bootstrapped
            _sub = "$_{CRD}$"
            range_bts_netzeroflux = rectangle(ax=ax,
                                              rect_lower_left_x=bts_zerocrossings_aggs['x_min'],
                                              rect_lower_left_y=0,
                                              rect_width=bts_zerocrossings_aggs['x_max'] - bts_zerocrossings_aggs[
                                                  'x_min'],
                                              rect_height=1,
                                              label=f"THR{_sub} range ({num_bootstrap_runs} bootstraps)",
                                              color=COLOR_THRESHOLD)

        ## Vertical line showing xCRD threshold
        line_xcrd_vertical = None
        if showline_xcrd:
            thres_xcrd = self.results_threshold_detection['thres_xcrd']
            num_xcrds = self.results_threshold_detection['num_xcrds']
            _sub = "$_{XCRD}$"
            line_xcrd_vertical = ax.axvline(thres_xcrd, lw=3, color=COLOR_THRESHOLD2, ls='-', zorder=99,
                                            label=f"THR{_sub}, VPD = {thres_xcrd:.2f} {self.x_units}")
            # t = ax.text(thres_xcrd + thres_xcrd * 0.02, 0.04, f"{num_xcrd} extremely critical days",
            #             size=INFOTXT_FONTSIZE,
            #             color='black', backgroundcolor='none', transform=ax.get_xaxis_transform(),
            #             alpha=1, horizontalalignment='left', verticalalignment='top', zorder=99)



        # Format
        ax.axhline(0, lw=1, color='black')
        xlabel = f"Daily maximum VPD (${self.x_units}$)"

        ylabel = f"{self.y_col} (${self.y_units}$)"
        plotfuncs.default_format(ax=ax, txt_xlabel=xlabel, txt_ylabel=ylabel)

        # Custom legend
        # Assign two of the handles to the same legend entry by putting them in a tuple
        # and using a generic handler map (which would be used for any additional
        # tuples of handles like (p1, p3)).
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
        l = ax.legend(
            [
                line_crd_vertical if showline_crd else None,
                range_bts_netzeroflux if showrange_crd else None,
                line_xcrd_vertical if showline_xcrd else None,
                line_xy_y,
                line_highlight if line_highlight else None,
                line_fit_y if showfit else None,
                line_fit_ci_y if showfit else None,
                line_fit_pb_y if showfit else None
            ],
            [
                line_crd_vertical.get_label() if showline_crd else None,
                range_bts_netzeroflux.get_label() if showrange_crd else None,
                line_xcrd_vertical.get_label() if showline_xcrd else None,
                line_xy_y.get_label(),
                line_highlight.get_label() if line_highlight else None,
                line_fit_y.get_label() if showfit else None,
                line_fit_ci_y.get_label() if showfit else None,
                line_fit_pb_y.get_label() if showfit else None
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
                size=theme.AXLABELS_FONTSIZE, color='black', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='left', verticalalignment='top')

    def _resample_dataset(self, df: DataFrame, aggs: list, day_start_hour: int = None):
        """Resample to daily values from *day_start_hour* to *day_start_hour*"""
        df_aggs = df.resample('D', offset=f'{day_start_hour}H').agg(aggs)
        df_aggs = df_aggs.where(df_aggs[self.x_col]['count'] == 48).dropna()  # Full days only
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

    def _bts_zerocrossings_collect(self, bts_results):
        bts_linecrossings_df = pd.DataFrame()
        for bts in range(1, self.bootstrap_runs + 1):  # +1 because 0 = non-bootstrapped run
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
                        y_col: str,
                        y_agg: str,
                        fit_to_bins: int = 10,
                        detect_zerocrossing: bool = False) -> dict:
        """Bootstrap ycols and fit to x"""

        # Get column names in aggregated df
        x_col = (x_col, x_agg)
        y_col = (y_col, y_agg)
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

                # Zero crossing for y
                if detect_zerocrossing:
                    zerocrossing_vals = \
                        self._detect_zerocrossing_y(fit_results=cur_fit_results['fit_df'])

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

            # Store bootstrap results in dict
            bts_results[bts] = fit_results
            bts += 1

        return bts_results

    def _detect_zerocrossing_y(self, fit_results: dict):
        # kudos: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value

        num_zerocrossings = None
        zerocrossings_ix = None

        # Collect predicted vals in df
        zerocrossings_df = pd.DataFrame()
        zerocrossings_df['x_col'] = fit_results['fit_x']
        zerocrossings_df['y_nom'] = fit_results['nom']

        # Check values above/below zero
        _signs = np.sign(zerocrossings_df['y_nom'])
        _signs_max = _signs.max()
        _signs_min = _signs.min()
        _signs_num_totalvals = len(_signs)
        _signs_num_abovezero = _signs.loc[_signs > 0].count()
        _signs_num_belowzero = _signs.loc[_signs < 0].count()

        if _signs_max == _signs_min:
            print("y does not cross zero-line.")
        else:
            zerocrossings_ix = np.argwhere(np.diff(_signs)).flatten()
            num_zerocrossings = len(zerocrossings_ix)

        # There must be one single line crossing to accept result
        # If there is more than one line crossing, reject result
        zerocrossings_ix = zerocrossings_ix[-1]  # Use last detected crossing
        # if num_zerocrossings > 1:
        #     return None

        # Values at zero crossing needed
        # Found index is last element *before* zero-crossing, therefore + 1
        zerocrossing_vals = zerocrossings_df.iloc[zerocrossings_ix + 1]
        zerocrossing_vals = zerocrossing_vals.to_dict()

        # NEE at zero crossing must zero or above,
        # i.e. reject if NEE after zero-crossing does not change to emission
        if (zerocrossing_vals['y_nom'] < 0):
            return None

        # x value must be above threshold to be somewhat meaningful, otherwise reject result
        if (zerocrossing_vals['x_col'] < 1):  # TODO currently hardcoded for VPD kPa
            # VPD is too low, must be at least 1 kPa for valid crossing
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
                     **kwargs):
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self.plot_crd_detection_results(ax=ax, **kwargs)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)


if __name__ == '__main__':
    pd.options.display.width = None
    pd.options.display.max_columns = None
    # pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)

    # Test data
    from diive.core.io.files import load_pickle

    df_orig = load_pickle(
        filepath=r'L:\Dropbox\luhk_work\_current\fp2022\7-14__IRGA627572__addingQCF0\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle')

    # Settings
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    subset_cols = [vpd_col, nee_col]
    df = df_orig[subset_cols].copy().dropna()
    # Convert units
    df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df = df.loc[df.index.year < 2022]
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)]

    # Critical days
    crd = CriticalDays(
        df=df,
        x_col=vpd_col,
        x_units='kPa',
        x_agg='max',
        y_col=nee_col,
        y_units="gCO_{2}\ m^{-2}\ d^{-1}",
        y_agg='sum',
        usebins=0,
        bootstrap_runs=3,
        bootstrapping_random_state=None
    )

    crd.detect_crd_threshold()
    results = crd.results_threshold_detection
    crd.plot_results(showfit=True,
                     highlight_year=2019,
                     showline_crd=True,
                     showrange_crd=True,
                     showline_xcrd=True)
