from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame, Series

from diive.core.dfun.frames import flatten_multiindex_all_df_cols
from diive.core.plotting.heatmap_xyz import HeatmapPivotXYZ
from diive.core.plotting.plotfuncs import default_format, save_fig
from diive.core.plotting.styles.LightTheme import COLOR_NEP, FONTSIZE_LEGEND
from diive.pkgs.analyses.quantilexyaggz import QuantileXYAggZ
from diive.pkgs.fits._fitter import QuadraticFit


class FluxCriticalHeatDaysP95:
    exception = f'Result not available, call .run() first.'
    n_bins = 20

    def __init__(self,
                 df: DataFrame,
                 ta_col: str,
                 vpd_col: str,
                 flux_col: str,
                 ta_agg: Literal['min', 'max', 'mean', 'median', 'sum'] = 'max',
                 vpd_agg: Literal['min', 'max', 'mean', 'median', 'sum'] = 'max',
                 flux_agg: Literal['min', 'max', 'mean', 'median', 'sum'] = 'sum',
                 additional_cols: list = None):
        """
        Calculate the thresholds for air temperature (TA) and vapor
        pressure deficit (VPD) for critical heat days, defined as days
        when the respective TA and VPD daily maxima (default) were
        >= their respective 95th percentile.

        Data *df* are converted to daily aggregates. By default, for TA
        the daily maxima are calculated, for VPD the daily maxima and
        for the flux the daily sums.

        The aggregated values for TA and VPD are then divided into
        20 equally-sized bins, respectively. The 20th bin contains all
        data in the 95th - 100th percentile range. The thresholds
        corresponds to the starting value of the 95th bin, respectively
        for TA and VPD. Days when both the daily maximum TA and the
        daily maximum VPD equal or exceed this threshold are labelled
        as critical heat days.

        chd / CHD = critical heat days
        nchd / nCHD = near-critical heat days

        Args:
            df: Input data as time series at time resolution < 1 day
                For this analysis the time series are later aggregated
                to daily values.
            ta_col: Name of the column in *df* that contains the air
                temperature time series
            vpd_col: Name of the column in *df* that contains the
                VPD time series
            flux_col: Name of the column in *df* that contains the flux
                time series
            ta_agg: Aggregation method for air temperature from which the
                threshold is calculated. If the default 'max' is used, then
                the air temperature threshold for CHD corresponds to the
                95th percentile of daily max air temperatures.
            vpd_agg: Aggregation method for VPD from which the threshold
                is calculated. If the default 'max' is used, then the
                VPD threshold for CHD corresponds to the 95th percentile
                of daily max VPD values.
            flux_agg: # todo
            additional_cols: List of additional variables (column names)
                that should be included in some output results. This makes
                sense e.g. for other fluxes, such as GPP and RECO.
        """
        self.df = df
        self.ta_col = ta_col
        self.vpd_col = vpd_col
        self.flux_col = flux_col
        self.ta_agg = ta_agg
        self.vpd_agg = vpd_agg
        self.flux_agg = flux_agg
        self.additional_cols = additional_cols

        # Data series
        self.ta_agg_col = f"{self.ta_col}_{self.ta_agg}"
        self.vpd_agg_col = f"{self.vpd_col}_{self.vpd_agg}"
        self.flux_agg_col = f"{self.flux_col}_{self.flux_agg}"

        # Instance attributes
        self._xdata = None
        self._ydata = None
        self._zdata = None
        self._xyz_pivot_df = None
        self._xyz_long_df = None

        self._xyz_long_extended_df = None
        self._combobins_df = None
        self._xyz_long_extended_bins_equal_df = None
        self._xyz_long_extended_bins_vpdhigher_df = None
        self._xyz_long_extended_bins_tahigher_df = None
        self._combobins_bins_equal_df = None
        self._combobins_bins_tahigher_df = None
        self._combobins_bins_vpdhigher_df = None
        self._thres_chd_ta = None
        self._thres_chd_vpd = None
        self._thres_nchd_ta = None
        self._thres_nchd_vpd = None
        self._fit_results = None
        self._xyz_long_extended_criticalheatdays_df = None
        self._xyz_long_extended_nearcriticalheatdays_df = None

        self._results = None

    def get_results(self) -> dict:
        """All results collected in dict"""
        if not isinstance(self._results, dict):
            raise Exception(self.exception)
        return self._results

    def _checkprop(self, p, dtype):
        if not isinstance(p, dtype):
            raise Exception(self.exception)
        return p

    @property
    def xdata(self) -> Series:
        """Time series of daily aggregated x values"""
        return self._checkprop(self._xdata, Series)

    @property
    def ydata(self) -> Series:
        """Time series of daily aggregated y values"""
        return self._checkprop(self._ydata, Series)

    @property
    def zdata(self) -> Series:
        """Time series of daily aggregated z values"""
        return self._checkprop(self._zdata, Series)

    @property
    def thres_chd_ta(self) -> float:
        """Threshold for air temperature to define critical heat days"""
        return self._checkprop(self._thres_chd_ta, float)

    @property
    def thres_chd_vpd(self) -> float:
        """Threshold for VPD to define critical heat days"""
        return self._checkprop(self._thres_chd_vpd, float)

    @property
    def thres_nchd_ta(self) -> tuple:
        """Upper and lower thresholds for air temperature to define near-critical heat days"""
        return self._checkprop(self._thres_nchd_ta, tuple)

    @property
    def thres_nchd_vpd(self) -> tuple:
        """Upper and lower thresholds for VPD to define near-critical heat days"""
        return self._checkprop(self._thres_nchd_vpd, tuple)

    @property
    def xyz_pivot_df(self) -> DataFrame:
        """Aggregated z in xy quantiles, pivoted"""
        return self._checkprop(self._xyz_pivot_df, DataFrame)

    @property
    def xyz_long_df(self) -> DataFrame:
        """Time series of xyz, including binning info"""
        return self._checkprop(self._xyz_long_df, DataFrame)

    @property
    def xyz_long_extended_df(self) -> DataFrame:
        """Time series of xyz with additional aggregates and including
        extended binning info"""
        return self._checkprop(self._xyz_long_extended_df, DataFrame)

    @property
    def xyz_long_extended_bins_equal_df(self) -> DataFrame:
        """Same data as *xyz_long_extended_df*, but only containing data
        where x and y fall into the same respective bin, e.g. both are in
        their respective 90th percentile"""
        return self._checkprop(self._xyz_long_extended_bins_equal_df, DataFrame)

    @property
    def xyz_long_extended_bins_tahigher_df(self) -> DataFrame:
        """Same data as *xyz_long_extended_df*, but only containing data
        where x falls into a higher bin than y, e.g. x falls into the 60th
        percentile and y falls into the respective 45th percentile"""
        return self._checkprop(self._xyz_long_extended_bins_tahigher_df, DataFrame)

    @property
    def xyz_long_extended_bins_vpdhigher_df(self) -> DataFrame:
        """Same data as *xyz_long_extended_df*, but only containing data
        where y falls into a higher bin than x, e.g. y falls into the 80th
        percentile and x falls into the respective 50th percentile"""
        return self._checkprop(self._xyz_long_extended_bins_vpdhigher_df, DataFrame)

    @property
    def combobins_df(self) -> DataFrame:
        """Aggregated xyz in combined bins (sum) of x and y. The combined bins
        can result from any combination of x and y bins, e.g. the combined bin
        100 can result from the combination 60+40 (x in the 60th percentile bin,
        y in the 40th percentile bin), but also from any other combination that
        results in a sum of 100 (e.g. 30+70, 80+20, 50+50, ...). However, the most
        extreme combobin 190 has only one possible combination (95+95), which is
        also the case for combobin 0 (0+0)."""
        return self._checkprop(self._combobins_df, DataFrame)

    @property
    def combobins_bins_equal_df(self) -> DataFrame:
        """Aggregated xyz in combo bins, but only including z values
        where x and y are members of the same respective bin, e.g. both x and
        y fall into their respective 65th percentile bin. The resulting dataframe
        shows the sum of the combined bins, e.g. the combined bin 100 collects
        data where both x and y were in their respective 50th percentile bin,
        (50+50), e.g. combined bin 190 is the most extreme combo (95+95).
        See also docstring for *combobins_df*."""
        return self._checkprop(self._combobins_bins_equal_df, DataFrame)

    @property
    def combobins_bins_tahigher_df(self) -> DataFrame:
        """Aggregated xyz in combo bins, but only including z values
        where x falls into a higher respective bin than and y.
        See also docstring for *combobins_df*."""
        return self._checkprop(self._combobins_bins_tahigher_df, DataFrame)

    @property
    def combobins_bins_vpdhigher_df(self) -> DataFrame:
        """Aggregated xyz in combo bins, but only including z values
        where y falls into a higher respective bin than and x.
        See also docstring for *combobins_df*."""
        return self._checkprop(self._combobins_bins_vpdhigher_df, DataFrame)

    @property
    def xyz_long_extended_criticalheatdays_df(self) -> DataFrame:
        """Time series of xyz with additional aggregates and including
        extended binning info, but only including data where x and y
        are above their respective threshold.
        See also docstring for *xyz_long_extended_df*."""
        return self._checkprop(self._xyz_long_extended_criticalheatdays_df, DataFrame)

    @property
    def xyz_long_extended_nearcriticalheatdays_df(self) -> DataFrame:
        """Time series of xyz with additional aggregates and including
        extended binning info, but only including data where (1) x and y
        are both in their respective 90th percentile bin, or where (2) x
        or y are in their respective 95th bin, while the other is in the
        90th bin. Therefore, the following percentile bin combinations define
        near-extreme conditions: 90+90, 95+90 or 90+95."""
        return self._checkprop(self._xyz_long_extended_nearcriticalheatdays_df, DataFrame)

    @property
    def fit_results(self) -> dict:
        """XXX"""
        return self._checkprop(self._fit_results, dict)

    def _setnames(self, df: DataFrame):
        """Get data for xyz with desired aggregation"""
        x = df[self.ta_agg_col].copy()
        y = df[self.vpd_agg_col].copy()
        z = df[self.flux_agg_col].copy()
        return x, y, z

    def run(self, bins_min_n_vals: int = 5, verbose: bool = True):

        # Create xyz subset
        _subsetdf = self._create_subset()

        # Calc xyz daily aggregates
        _subsetdf_daily = self._resample_daily_aggs(subset=_subsetdf)

        # Get data for xyz
        self._xdata, self._ydata, self._zdata = self._setnames(df=_subsetdf_daily)

        # Based on the x and y aggregates, assign bins to data and then aggregate z in bins
        self._xyz_pivot_df, self._xyz_long_df = self._assign_bins(binagg_z='mean',
                                                                  n_quantiles=self.n_bins,
                                                                  min_n_vals_per_bin=bins_min_n_vals)

        # Collect bin info and add it to the extended dataframe
        _bin_info = self._collect_bininfo(df=self._xyz_long_df)
        self._xyz_long_extended_df = self._add_bininfo(left=_subsetdf_daily, right=_bin_info)
        _binmissing = self._xyz_long_extended_df['BINS_COMBINED_INT'].isnull()
        self._xyz_long_extended_df = self._xyz_long_extended_df[~_binmissing].copy()

        # Calculate difference between xy bins
        self._xyz_long_extended_df['BIN_DIFF'] = self._bin_difference()

        # Collect data for different bin scenarios
        # (1) Data where TA and VPD are in the same bin, respectively, e.g. both in 50th quantile
        self._xyz_long_extended_bins_equal_df = self.xyz_long_extended_df.loc[
            self.xyz_long_extended_df['BIN_DIFF'] == 0].copy()

        # (2) Data where TA is in a higher bin than VPD
        self._xyz_long_extended_bins_tahigher_df = self.xyz_long_extended_df.loc[
            self.xyz_long_extended_df['BIN_DIFF'] > 5].copy()

        # (3) Data where VPD is in a higher bin than TA
        self._xyz_long_extended_bins_vpdhigher_df = self.xyz_long_extended_df.loc[
            self.xyz_long_extended_df['BIN_DIFF'] < 5].copy()

        # Calculate stats for combo bins for each bin scenario 1-3 and over all data
        self._combobins_df = self._calc_combobins(df=self.xyz_long_extended_df)
        self._combobins_bins_equal_df = self._calc_combobins(df=self.xyz_long_extended_bins_equal_df)
        self._combobins_bins_tahigher_df = self._calc_combobins(df=self.xyz_long_extended_bins_tahigher_df)
        self._combobins_bins_vpdhigher_df = self._calc_combobins(df=self.xyz_long_extended_bins_vpdhigher_df)

        # Collect extreme days
        self._xyz_long_extended_criticalheatdays_df, \
            self._xyz_long_extended_nearcriticalheatdays_df = \
            self._criticalheatdays_subset()

        # Get thresholds from pivot
        self._thres_chd_ta, \
            self._thres_chd_vpd, \
            self._thres_nchd_ta, \
            self._thres_nchd_vpd = \
            self._thresholds()

        # Collect results in dict
        self._results = self._collect_results()

        if verbose:
            self._results_overview()

    def _results_overview(self):
        res = self.get_results()
        print("# Results are stored in a dictionary that can be accessed with .get_results()."
              "\nDictionary keys:")
        for k in res.keys():
            print(f"  {k}")
        print("\n# Critical heat days are defined by:")
        print(f"  Daily maximum air temperature >= {res['thres_chd_ta']}")
        print(f"  Daily maximum VPD >= {res['thres_chd_vpd']}")

        thres_nchd_ta_lower = res['thres_nchd_ta'][0]
        thres_nchd_ta_upper = res['thres_nchd_ta'][1]
        thres_nchd_vpd_lower = res['thres_nchd_vpd'][0]
        thres_nchd_vpd_upper = res['thres_nchd_vpd'][1]
        print("\n# Near-critical heat days are defined by:")
        print(f"  Daily maximum air temperature >= {thres_nchd_ta_lower:.3f} and <= {thres_nchd_ta_upper:.3f}")
        print(f"  Daily maximum VPD >= {thres_nchd_vpd_lower:.3f} and <= {thres_nchd_vpd_upper:.3f}")

        print("\n# Number of critical and near-critical heat days:")
        print(f"  Number of critical heat days: {len(res['xyz_long_extended_criticalheatdays_df'])}")
        print(f"  Number of near-critical heat days: {len(res['xyz_long_extended_nearcriticalheatdays_df'])}")

    def _bin_difference(self) -> Series:
        """Calculate the difference between the bins"""
        xbins = f'BIN_{self.ta_agg_col}'
        ybins = f'BIN_{self.vpd_agg_col}'
        return self._xyz_long_extended_df[xbins] - self._xyz_long_extended_df[ybins]

    def _collect_results(self):
        return {
            'xdata': self.xdata,
            'ydata': self.ydata,
            'zdata': self.zdata,
            'thres_chd_ta': self.thres_chd_ta,
            'thres_chd_vpd': self.thres_chd_vpd,
            'thres_nchd_ta': self.thres_nchd_ta,
            'thres_nchd_vpd': self.thres_nchd_vpd,
            'xyz_pivot_df': self.xyz_pivot_df,
            'xyz_long_df': self.xyz_long_df,
            'xyz_long_extended_df': self.xyz_long_extended_df,
            'xyz_long_extended_bins_equal_df': self.xyz_long_extended_bins_equal_df,
            'xyz_long_extended_bins_tahigher_df': self.xyz_long_extended_bins_tahigher_df,
            'xyz_long_extended_bins_vpdhigher_df': self.xyz_long_extended_bins_vpdhigher_df,
            'xyz_long_extended_criticalheatdays_df': self.xyz_long_extended_criticalheatdays_df,
            'xyz_long_extended_nearcriticalheatdays_df': self.xyz_long_extended_nearcriticalheatdays_df,
            'combobins_df': self.combobins_df,
            'combobins_bins_equal_df': self.combobins_bins_equal_df,
            'combobins_bins_tahigher_df': self.combobins_bins_tahigher_df,
            'combobins_bins_vpdhigher_df': self.combobins_bins_vpdhigher_df,
        }

    def _criticalheatdays_subset(self):
        """Collect daily data of critical and near-critical heat days in separate subsets"""

        bin_ta = self.xyz_long_extended_df[f'BIN_{self.ta_agg_col}']
        bin_vpd = self.xyz_long_extended_df[f'BIN_{self.vpd_agg_col}']

        # Data for critical heat days
        # Both daily max (default) TA and VPD are in the 95-100th percentile range
        locs_chd = (bin_ta == 95) & (bin_vpd == 95)
        criticalheatdays_df = self.xyz_long_extended_df[locs_chd].copy()

        # Data for near-critical heat days
        # Both daily max (default) TA and VPD are in the 90-95th percentile range
        locs_nchd = (bin_ta == 90) & (bin_vpd == 90)
        nearcriticalheatdays_df = self.xyz_long_extended_df[locs_nchd].copy()

        criticalheatdays_df = criticalheatdays_df.set_index('DATE')
        nearcriticalheatdays_df = nearcriticalheatdays_df.set_index('DATE')
        criticalheatdays_df.index = pd.to_datetime(criticalheatdays_df.index)
        nearcriticalheatdays_df.index = pd.to_datetime(nearcriticalheatdays_df.index)

        return criticalheatdays_df, nearcriticalheatdays_df

    def _thresholds(self) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
        """Get the threshold values for critical and near-critical heat days"""

        # Thresholds for critical heat days
        thres_chd_ta = self.xyz_long_extended_criticalheatdays_df[self.ta_agg_col].min()
        thres_chd_vpd = self.xyz_long_extended_criticalheatdays_df[self.vpd_agg_col].min()

        # Thresholds for near-critical heat days
        thres_nchd_ta = self.xyz_long_extended_nearcriticalheatdays_df[self.ta_agg_col].min(), \
            self.xyz_long_extended_nearcriticalheatdays_df[self.ta_agg_col].max()
        thres_nchd_vpd = self.xyz_long_extended_nearcriticalheatdays_df[self.vpd_agg_col].min(), \
            self.xyz_long_extended_nearcriticalheatdays_df[self.vpd_agg_col].max()
        return thres_chd_ta, thres_chd_vpd, thres_nchd_ta, thres_nchd_vpd

    def _calc_combobins(self, df: DataFrame):
        dailybinned_df = df.groupby('BINS_COMBINED_INT').agg({
            self.flux_agg_col: ['mean', 'std', 'count'],
            self.ta_agg_col: ['min', 'max'],
            self.vpd_agg_col: ['min', 'max']
        })
        dailybinned_df[(self.flux_agg_col, 'mean+std')] = \
            dailybinned_df[self.flux_agg_col]['mean'].add(dailybinned_df[self.flux_agg_col]['std'])
        dailybinned_df[(self.flux_agg_col, 'mean-std')] = \
            dailybinned_df[self.flux_agg_col]['mean'].sub(dailybinned_df[self.flux_agg_col]['std'])
        return dailybinned_df

    def fit(self, n_predictions: int = 1000):

        _df = self.xyz_long_extended_bins_equal_df
        fitter = QuadraticFit(
            df=_df,
            xcol='BINS_COMBINED_INT',
            ycol=self.flux_agg_col,
            predict_max_x=_df['BINS_COMBINED_INT'].max(),
            predict_min_x=_df['BINS_COMBINED_INT'].min(),
            n_predictions=n_predictions
        )
        fitter.fit()
        self._fit_results = fitter.fit_results

    def showplot_heatmap_percentiles(self,
                                     saveplot: bool = False,
                                     title: str = None,
                                     path: Path or str = None,
                                     dpi: int = 72,
                                     **kwargs):
        fig = plt.figure(figsize=(10, 9), dpi=dpi)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self.plot_heatmap_percentiles(ax=ax, **kwargs)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)
        return ax

    def plot_heatmap_percentiles(self, ax, **kwargs):
        hm = HeatmapPivotXYZ(pivotdf=self.xyz_pivot_df)
        hm.plot(ax=ax,
                xlabel=r'Daily maximum air temperature ($\mathrm{percentile}$)',
                ylabel=r'Daily maximum VPD ($\mathrm{percentile}$)',
                zlabel=r'Net ecosystem production ($\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$)',
                tickpos=list(np.arange(-2.5, 102.5, 10)),
                ticklabels=list(range(0, 105, 10)),
                **kwargs)
        return ax

    def showplot_criticalheat(self,
                              saveplot: bool = False,
                              title: str = None,
                              path: Path or str = None,
                              dpi: int = 72,
                              **kwargs):
        fig = plt.figure(figsize=(9, 9), dpi=dpi)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self.plot_bins(ax=ax, **kwargs)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)
        return ax

    def plot_bins(self,
                  ax,
                  show_fit: bool = True,
                  show_prediction_interval: bool = True,
                  show_sd: bool = True,
                  xlabel: str = None,
                  ylabel: str = None,
                  xunits: str = None,
                  yunits: str = None):

        # # Difference
        # diff = DataFrame()
        # diff['a'] = dailybinned_df[self.zvaragg]['mean']
        # diff['aa'] = diff['a'].shift(1)
        # diff['aaa'] = diff['a'].sub(diff['aa'])
        # diff['aaa'].plot()
        # plt.show()

        _plotdf = self.combobins_bins_equal_df

        # Plot z var as y in scatter plot
        _label = xlabel if xlabel else self.flux_col
        _n_vals_min = _plotdf[self.flux_agg_col]['count'].min()
        _n_vals_max = _plotdf[self.flux_agg_col]['count'].max()
        _label = f"{_label} (between {_n_vals_min} and {_n_vals_max} days per bin)"
        _y = _plotdf[self.flux_agg_col]['mean']
        # _y.index = _y.index / 2
        line_xy = ax.scatter(_y.index, _y,
                             edgecolor='none', color=COLOR_NEP, alpha=1,
                             s=150, label=_label, zorder=99, marker='o')

        # # Plot z var as y in scatter plot
        # # _label = xlabel if xlabel else self.zvar
        # _y = dailybinned_df[self.xvaragg]['min']
        # xxx = ax.scatter(_y.index, _y,
        #                  edgecolor='none', color='blue', alpha=1,
        #                  s=100, label='XXX', zorder=99, marker='o')
        # _y = dailybinned_df[self.yvaragg]['min'].multiply(20)
        # xxx = ax.scatter(_y.index, _y,
        #                  edgecolor='none', color='green', alpha=1,
        #                  s=100, label='XXX', zorder=99, marker='o')

        if show_sd:
            _upper = _plotdf[self.flux_agg_col]['mean+std']
            _lower = _plotdf[self.flux_agg_col]['mean-std']
            line_sd = ax.fill_between(_upper.index, _upper, _lower,
                                      alpha=.1, zorder=0, color=COLOR_NEP,
                                      edgecolor='none', label="standard deviation")

        # Fit and confidence intervals
        line_fit = line_fit_ci = None
        line_fit_pb = None
        if show_fit:
            line_fit = self._plot_fit(ax=ax, color_fitline='red')
            line_fit_ci = self._plot_fit_ci(ax=ax, color_fitline='red')

        # Prediction bands
        if show_prediction_interval:
            line_fit_pb = self._plot_fit_pi(ax=ax, color_fitline='blue')

        # # Rectangle
        # _s = self.dailybinneddf[self.xvaragg]['min']
        # _locs = (_s > 15) & (_s < 20)
        # _s = list(_s[_locs].index)
        # import numpy as np
        # _start = np.min(_s)
        # _width = np.max(_s) - _start
        # range_bts_netzeroflux = rectangle(
        #     ax=ax,
        #     rect_lower_left_x=_start,
        #     rect_lower_left_y=.9,
        #     rect_width=_width,
        #     rect_height=.2,
        #     label=f"xxx",
        #     color='blue')

        # Axes labels
        ax.axhline(0, lw=1, color='black')
        if xlabel:
            xlabel = f"{xlabel} ({xunits})"
        else:
            xlabel = f"Combined percentiles of daily {self.ta_agg} {self.ta_col} " \
                     f"and daily {self.vpd_agg} {self.vpd_col}"
        if ylabel:
            ylabel = f"{ylabel} ({yunits})"
        else:
            ylabel = f"{self.flux_col} ({yunits})"

        # Format
        default_format(ax=ax, showgrid=False, ax_xlabel_txt=xlabel, ax_ylabel_txt=ylabel)
        _min = _plotdf[self.flux_agg_col]['mean-std'].min()
        _max = _plotdf[self.flux_agg_col]['mean+std'].max()
        ax.set_ylim([_min, _max])

        # Zero line
        ax.axhline(0, color='black', lw=1)

        # specify the number of ticks on the x-axis
        ax.locator_params(axis='y', nbins=20)
        _range = range(0, 210, 20)
        ax.set_xticks(_range)

        # Custom legend
        # This legend replaces the legend from PlotBinFitterBTS
        # Assign two of the handles to the same legend entry by putting them in a tuple
        # and using a generic handler map (which would be used for any additional
        # tuples of handles like (p1, p3)).
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
        l = ax.legend(
            [
                # line_dcrit_vertical if showline_dcrit else None,
                # range_bts_netzeroflux if showrange_dcrit else None,
                line_xy,
                line_sd,
                # line_highlight if line_highlight else None,
                line_fit if show_fit else None,
                line_fit_ci if show_fit else None,
                # line_fit_pb if show_fit else None
            ],
            [
                # line_dcrit_vertical.get_label() if showline_dcrit else None,
                # range_bts_netzeroflux.get_label() if showrange_dcrit else None,
                line_xy.get_label(),
                line_sd.get_label() if show_sd else None,
                # line_highlight.get_label() if line_highlight else None,
                line_fit.get_label() if show_fit else None,
                line_fit_ci.get_label() if show_fit else None,
                # line_fit_pb.get_label() if show_fit else None
            ],
            bbox_to_anchor=(0, 1),
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
            loc="lower left",
            ncol=2,
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)})

        return ax

    # t = []
    # for bts_run in range(1, 10):
    #     sample_df = self.subset.sample(n=int(len(self.subset)), replace=True)
    #     z_pivot, z_long = self._assign_bins(df=sample_df, binagg_z='mean')
    #     bin_info = self._collect_bininfo(df=z_long)
    #     _subset_bin_info = self._add_bininfo(left=sample_df, right=bin_info)

    # xx = _subset_bin_info.loc[_subset_bin_info['BINS_COMBINED_INT'] == 200]['VPD_f_max'].max()
    # t.append(xx)

    # d = DataFrame()
    # d['MEAN'] = _subset_bin_info.groupby('BINS_COMBINED_INT').mean()['NEP_mean']
    # d['SD'] = _subset_bin_info.groupby('BINS_COMBINED_INT').std()['NEP_mean']
    # d['MEAN+SD'] = d['MEAN'] + d['SD']
    # d['MEAN-SD'] = d['MEAN'] - d['SD']
    # plt.scatter(d.index, d['MEAN'])
    # plt.fill_between(d.index.values, d['MEAN+SD'].values, d['MEAN-SD'].values,
    #                  alpha=.1, zorder=0, color='black', edgecolor='none', label="mean±1sd")
    # plt.axhline(0)
    # plt.tight_layout()
    # plt.show()
    #
    # print(np.median(t))

    # extreme_ix = z_long.loc[z_long['BINS_COMBINED_INT'] == 200]['index']
    # extreme_ix = list(extreme_ix)
    #
    # a = z_long.loc[z_long['index'].isin(extreme_ix)]
    # plt.scatter(a['Tair_f_max'], a['NEP_mean'], c=a['VPD_f_max'])
    # plt.scatter(a['VPD_f_max'], a['NEP_mean'], c=a['Tair_f_max'])
    # plt.show()
    #
    # is_daytime = self.df['Rg_f'] > 50
    # daytime = self.df[is_daytime].copy()
    # daytime['DATE'] = daytime.index.date
    #
    # a = daytime.loc[daytime['DATE'].isin(extreme_ix)]
    # plt.scatter(a['Tair_f'], a['NEP'], c=a['VPD_f'])
    # plt.scatter(a['VPD_f'], a['NEP'], c=a['Tair_f'])
    # plt.show()

    @staticmethod
    def _collect_bininfo(df: DataFrame):
        """Collect columns containig bin info with timestamp"""
        cols = ['DATE']
        [cols.append(c) for c in df.columns if str(c).startswith('BIN')]
        bin_info = df[cols].copy()
        return bin_info

    @staticmethod
    def _add_bininfo(left: DataFrame, right: DataFrame) -> DataFrame:
        _subset = left.copy()
        _subset = _subset.reset_index(drop=False).copy()
        subset_bin_info = _subset.merge(right, left_on='DATE', right_on='DATE', how='left')
        return subset_bin_info

    def _create_subset(self):
        subset = self.df[[self.ta_col, self.vpd_col, self.flux_col]].copy()
        _additional_data = self.df[self.additional_cols].copy()
        subset = pd.concat([subset, _additional_data], axis=1)
        # Remove duplicates, each variable is only needed once
        subset = subset.loc[:, ~subset.columns.duplicated()]
        return subset  # todo add gpp reco]].copy()

    @staticmethod
    def _resample_daily_aggs(subset: DataFrame):
        """Aggregation to daily values, the resulting multiindex is flattened"""
        _aggs = ['min', 'max', 'mean', 'median', 'sum', 'count']
        subset = subset.groupby(subset.index.date).agg(_aggs)
        subset = flatten_multiindex_all_df_cols(df=subset)
        subset.index.name = 'DATE'
        return subset

    def _assign_bins(self, **params):
        qua = QuantileXYAggZ(x=self._xdata, y=self._ydata, z=self._zdata, **params)
        qua.run()
        return qua.pivotdf, qua.longformdf

    def _plot_fit(self, ax, color_fitline):
        a = self.fit_results['fit_params_opt'][0]
        b = self.fit_results['fit_params_opt'][1]
        c = self.fit_results['fit_params_opt'][2]
        operator1 = "+" if b > 0 else ""
        operator2 = "+" if c > 0 else ""
        label_fit = rf"$y = {a:.4f}x^2{operator1}{b:.4f}x{operator2}{c:.4f}$"
        line_fit, = ax.plot(self.fit_results['fit_df']['fit_x'],
                            self.fit_results['fit_df']['nom'],
                            c=color_fitline, lw=3, zorder=99, alpha=1, label=label_fit)
        return line_fit

    def _plot_fit_ci(self, ax, color_fitline):
        # Fit confidence region
        # Uncertainty lines (95% confidence)
        line_fit_ci = ax.fill_between(self.fit_results['fit_df']['fit_x'],
                                      self.fit_results['fit_df']['nom_lower_ci95'],
                                      self.fit_results['fit_df']['nom_upper_ci95'],
                                      alpha=.2, color=color_fitline, zorder=1,
                                      label="95% confidence region")
        return line_fit_ci

    def _plot_fit_pi(self, ax, color_fitline):
        # Fit prediction interval
        # Lower prediction band (95% confidence)
        ax.plot(self.fit_results['fit_df']['fit_x'],
                self.fit_results['fit_df']['lower_predband'],
                color=color_fitline, ls='--', zorder=97, lw=2,
                label="95% prediction interval")
        # ax.fill_between(self.fit_results['fit_df']['fit_x'],
        #                 self.fit_results['fit_df']['bts_lower_predband_Q97.5'],
        #                 self.fit_results['fit_df']['bts_lower_predband_Q02.5'],
        #                 alpha=.2, color=color_fitline, zorder=97,
        #                 label="95% confidence region")
        # Upper prediction band (95% confidence)
        line_fit_pb, = ax.plot(self.fit_results['fit_df']['fit_x'],
                               self.fit_results['fit_df']['upper_predband'],
                               color=color_fitline, ls='--', zorder=96, lw=2,
                               label="95% prediction interval")
        # ax.fill_between(self.fit_results['fit_df']['fit_x'],
        #                 self.fit_results['fit_df']['bts_upper_predband_Q97.5'],
        #                 self.fit_results['fit_df']['bts_upper_predband_Q02.5'],
        #                 alpha=.2, color=color_fitline, zorder=97,
        #                 label="95% confidence region")
        return line_fit_pb


def example():
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    nee_orig = 'NEE_CUT_REF_orig'
    nep_col = 'NEP'
    ta_col = 'Tair_f'
    gpp_dt_col = 'GPP_DT_CUT_REF'
    reco_dt_col = 'Reco_DT_CUT_REF'
    gpp_nt_col = 'GPP_CUT_REF_f'
    reco_nt_col = 'Reco_CUT_REF'
    ratio_dt_gpp_reco = 'RATIO_DT_GPP_RECO'
    rh_col = 'RH'
    swin_col = 'Rg_f'

    # Load data, using pickle for fast loading
    from diive.core.io.files import load_pickle
    source_file = r"L:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\__manuscripts\11.01_NEP-Penalty_CH-DAV_1997-2022 (2023)\data\CH-DAV_FP2022.5_1997-2022_ID20230206154316_30MIN.diive.csv.pickle"
    df_orig = load_pickle(filepath=source_file)

    # Data between May and Sep
    df_orig = df_orig.loc[(df_orig.index.month >= 4) & (df_orig.index.month <= 10)].copy()

    # Subset
    df = df_orig[[nee_col, vpd_col, ta_col,
                  gpp_dt_col, reco_dt_col, gpp_nt_col, reco_nt_col,
                  swin_col]].copy()

    # Convert units
    df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[nep_col] = df[nee_col].multiply(-1)  # Convert NEE to NEP, net uptake is now positive
    df[gpp_dt_col] = df[gpp_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[gpp_nt_col] = df[gpp_nt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[reco_dt_col] = df[reco_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[reco_nt_col] = df[reco_nt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[ratio_dt_gpp_reco] = df[gpp_dt_col].divide(df[reco_dt_col])
    df['nepmodel'] = df[gpp_nt_col].sub(df[reco_nt_col])

    # #  todo TESTING
    # t = df[[gpp_dt_col, reco_dt_col, ta_col]].copy()
    # tt = t.groupby(t.index.date).sum()
    # tt['ratio'] = tt[gpp_dt_col].divide(tt[reco_dt_col])
    # ttt = tt.loc[tt['ratio'] < 100].copy()
    # import pandas as pd
    # labels = range(0, 20, 1)
    # group, bins = pd.qcut(ttt[ta_col],q=20,retbins=True,duplicates='drop', labels=labels)  # How awesome!
    # ttt['group'] = group
    # tttt = ttt.groupby('group').mean()
    # tttt['ratio'].plot()
    # plt.show()

    df = df.loc[df[ratio_dt_gpp_reco] < 10].copy()

    ta_col = ta_col
    vpd_col = vpd_col
    flux_col = nep_col

    ext = FluxCriticalHeatDaysP95(df=df,
                                  ta_col=ta_col, ta_agg='max',
                                  vpd_col=vpd_col, vpd_agg='max',
                                  flux_col=flux_col, flux_agg='sum')
    ext.run(bins_min_n_vals=10, verbose=True)
    ext.showplot_heatmap_percentiles(cb_digits_after_comma=0)
    res = ext.get_results()

    ext.fit(n_predictions=1000)
    print(ext.fit_results.keys())
    ext.showplot_criticalheat(show_fit=True,
                              show_prediction_interval=False,
                              show_sd=True,
                              yunits="$\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$")


if __name__ == '__main__':
    example()
