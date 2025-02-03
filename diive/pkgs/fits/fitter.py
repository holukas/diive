from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
from scipy import stats
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp

from diive.core.dfun.fits import groupagg
from diive.core.plotting.plotfuncs import default_legend, default_format, add_zeroline_y


class BinFitterCP:

    def __init__(
            self,
            df: pd.DataFrame,
            xcol: str or tuple,
            ycol: str or tuple,
            predict_max_x: float = None,
            predict_min_x: float = None,
            n_predictions: int = 1000,
            n_bins_x: int = 10,
            bins_y_agg: Literal['mean', 'median'] = 'mean',
            fit_type: Literal['linear', 'quadratic_offset', 'quadratic', 'cubic'] = 'quadratic_offset'
    ):
        """Fit function to (binned) data and give CI and bootstrapped PI.

        Args:
            df: Dataframe with data for variables *xcol* and *ycol*.
            xcol: Variable name in *df*, will be used for creating *n_bins_x* number of bins.
            ycol: Variable name in *df*, will be aggregated in x-bins using method *bins_y_agg*.
            predict_max_x: Maximum x to predict y.
            predict_min_x: Minimum x to predict y.
            n_predictions: Number of predictions, e.g. 1000 means y will be predicted for 1000 x-values.
            n_bins_x: Number of bins for y-aggregation. Bins are equally-sized, based on variable defined in *xcol*.
                If *n_bins_x=0*, then the fit is done for the unbinned data.
            bins_y_agg: Aggregation type for y in each x-bin. Options: 'mean', 'median'
            fit_type: Fit type to fit to the bins. Options: 'linear', 'quadratic_offset', 'quadratic', 'cubic'
        """
        self.df = df[[xcol, ycol]].dropna()  # Remove NaNs, working data
        self.xcol = xcol
        self.ycol = ycol
        self.x = self.df[self.xcol]
        self.y = self.df[self.ycol]
        self.len_y = len(self.y)
        self.bins_y_agg = bins_y_agg
        self.n_predictions = n_predictions
        self.usebins = n_bins_x if n_bins_x >= 0 else 0  # Must be positive
        # self.fit_x_max = predict_max_x if isinstance(predict_max_x, float) else self.x.max()
        # self.fit_x_min = predict_min_x if isinstance(predict_min_x, float) else self.x.min()
        self.n_predictions = n_predictions if isinstance(n_predictions, int) else len(self.x)
        self.n_predictions = 2 if self.n_predictions < 2 else self.n_predictions

        self.equation = self._set_fit_equation(eqtype=fit_type)
        self.fit_type = fit_type

        self.fit_results = {}  # Stores fit results

    def run(self):
        self.fit_results = self._fit()

    def get_results(self):
        return self.fit_results

    @staticmethod
    def _predband(px, x, y, params_opt, func, conf=0.95):
        """Prediction band"""
        # px = requested points, x = x data, y = y data, params_opt = parameters, func = function name
        alpha = 1.0 - conf  # significance
        N = x.size  # data sample size
        var_n = len(params_opt)  # number of parameters
        q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)  # Quantile of Student's t distribution for p=(1-alpha/2)
        se = np.sqrt(1. / (N - var_n) * np.sum((y - func(x, *params_opt)) ** 2))  # Stdev of an individual measurement
        sx = (px - x.mean()) ** 2  # Auxiliary definition
        sxd = np.sum((x - x.mean()) ** 2)  # Auxiliary definition
        yp = func(px, *params_opt)  # Predicted values (best-fit model)
        dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))  # Prediction band
        lpb, upb = yp - dy, yp + dy  # Upper & lower prediction bands.
        return lpb, upb

    def _set_fit_equation(self, eqtype: str = 'quadratic_offset'):
        if eqtype == 'quadratic_offset':
            equation = self._fit_quadratic_offset
        elif eqtype == 'quadratic':
            equation = self._fit_quadratic
        elif eqtype == 'cubic':
            equation = self._fit_cubic
        elif eqtype == 'linear':
            equation = self._fit_linear
        else:
            equation = self._fit_quadratic_offset
        return equation

    def _fit_linear(self, x, a, b):
        """Linear fit"""
        return a * x + b

    def _fit_quadratic_offset(self, x, a, b, c):
        """Quadratic equation"""
        return a * x ** 2 + b * x + c

    def _fit_cubic(self, x, a, b, c, d):
        """Cubic equation"""
        # ax3+bx2+cx+d = 0
        return a * x ** 3 + b * x ** 2 + c * x + d

    def _fit_quadratic(self, x, a, b):
        """Quadratic equation"""
        return a * x ** 2 + b * x

    def _set_fit_data(self, df):
        # Bin data
        n_vals_per_bin = {}
        if self.usebins == 0:
            _df = df.copy()
            x = self.x
            y = self.y
            len_y = len(y)
            n_vals_per_bin['min'] = len_y
            n_vals_per_bin['max'] = len_y
        else:
            _df = groupagg(df=df, num_bins=self.usebins, bin_col=self.xcol)
            _df = _df.dropna()
            x = _df[self.xcol][self.bins_y_agg]
            y = _df[self.ycol][self.bins_y_agg]
            len_y = len(self.y)
            n_vals_per_bin = \
                _df[self.ycol]['count'].describe()[['min', 'max']].to_dict()
        return _df, x, y, len_y, n_vals_per_bin

    def _fit(self):
        """Calculate curve fit, confidence intervals and prediction bands

        kudos: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
        """

        df, x, y, len_y, n_vals_per_bin = self._set_fit_data(df=self.df)


        # Fit function f to data
        fit_params_opt, fit_params_cov = curve_fit(self.equation, x, y)

        a = fit_params_opt[0]
        b = fit_params_opt[1]
        operator1 = "+" if b > 0 else ""

        fit_equation_str = None
        kwargs = None
        fit_x = np.linspace(x.min(), x.max(), self.n_predictions)
        fit_y = None

        if self.fit_type == 'linear':
            fit_equation_str = rf"y = {a:.4f}x{operator1}{b:.4f}"
            kwargs = dict(x=x, a=a, b=b)
            a, b = unc.correlated_values(fit_params_opt, fit_params_cov)
            fit_y = a * fit_x + b

        elif self.fit_type == 'quadratic_offset':
            c = fit_params_opt[2]
            operator2 = "+" if c > 0 else ""
            fit_equation_str = rf"y = {a:.4f}x^2{operator1}{b:.4f}x{operator2}{c:.4f}"
            kwargs = dict(x=x, a=a, b=b, c=c)
            a, b, c = unc.correlated_values(fit_params_opt, fit_params_cov)
            fit_y = a * fit_x ** 2 + b * fit_x + c

        elif self.fit_type == 'cubic':
            c = fit_params_opt[2]
            d = fit_params_opt[3]
            operator2 = "+" if c > 0 else ""
            operator3 = "+" if d > 0 else ""
            fit_equation_str = rf"y = {a:.4f}x^3{operator1}{b:.4f}x^2{operator2}{c:.4f}x{operator3}{d:.4f}"
            kwargs = dict(x=x, a=a, b=b, c=c, d=d)
            a, b, c, d = unc.correlated_values(fit_params_opt, fit_params_cov)
            fit_y = a * fit_x ** 3 + b * fit_x ** 2 + c * fit_x + d

        elif self.fit_type == 'quadratic':
            fit_equation_str = rf"y = {a:.4f}x^2{operator1}{b:.4f}x"
            kwargs = dict(x=x, a=a, b=b)
            a, b = unc.correlated_values(fit_params_opt, fit_params_cov)
            fit_y = a * fit_x ** 2 + b * fit_x

        # Calc r2
        fit_r2 = 1.0 - (sum((y - self.equation(**kwargs)) ** 2) / ((len_y - 1.0) * np.var(y, ddof=1)))

        # Calculate parameter confidence interval
        # Calculate regression confidence interval

        nom = unp.nominal_values(fit_y)
        std = unp.std_devs(fit_y)

        # Best lower and upper prediction bands
        lower_predband, upper_predband = \
            self._predband(px=fit_x, x=x, y=y,
                           params_opt=fit_params_opt, func=self.equation, conf=0.95)

        # Collect fit data (time series)
        fit_df = pd.DataFrame()
        fit_df['fit_x'] = fit_x
        fit_df['fit_y'] = fit_y
        fit_df['std'] = std
        fit_df['nom'] = nom
        fit_df['lower_predband'] = lower_predband
        fit_df['upper_predband'] = upper_predband

        # Calc 95% confidence region of fit, confidence interval
        fit_df['nom_lower_ci95'] = fit_df['nom'] - 1.96 * fit_df['std']
        fit_df['nom_upper_ci95'] = fit_df['nom'] + 1.96 * fit_df['std']

        # Collect results in dict
        fit_results = dict(input_df=self.df,
                           xvar=self.xcol,
                           yvar=self.ycol,
                           fit_df=fit_df,
                           fit_equation=self.equation,
                           fit_equation_str=fit_equation_str,
                           fit_type=self.fit_type,
                           fit_params_opt=fit_params_opt,
                           fit_params_cov=fit_params_cov,
                           fit_r2=fit_r2,
                           bin_df=df.copy(),
                           bins_x=x,
                           bins_y=y,
                           n_vals_per_bin=n_vals_per_bin)

        return fit_results

    def showplot(self,
                 ax=None,
                 show_unbinned_data: bool = True,
                 show_bin_variation: bool = True,
                 unbinned_alpha: float = .5,
                 highlight_year: int = None,
                 bin_size: int = 90,
                 bin_color: str = 'black',
                 bin_edgecolor: str = 'none',
                 bin_marker: str = 'o',
                 bin_alpha: float = 1,
                 fitline_color: str = 'red',
                 label: str = 'label',
                 showfit: bool = True,
                 show_prediction_interval: bool = True,
                 xlim: tuple = None,
                 ylim: tuple = None,
                 title: str = None,
                 xlabel: str = None,
                 ylabel: str = None
                 ):

        # Fitplot
        fig = plt.figure(facecolor='white', figsize=(9, 9), dpi=100)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0, hspace=0, left=.2, right=.8, top=.8, bottom=.2)
        ax = fig.add_subplot(gs[0, 0])

        # x/y
        _numvals_y = len(self.fit_results['bins_y'])
        n_vals_min = int(self.fit_results['n_vals_per_bin']['min'])
        n_vals_max = int(self.fit_results['n_vals_per_bin']['max'])
        _label = (f"{len(self.fit_results['bin_df'])} {self.bins_y_agg} bins "
                  f"({n_vals_min}-{n_vals_max} values per bin)")

        # Input data
        if show_unbinned_data:
            line_xy = ax.scatter(
                x=self.fit_results['input_df'][self.xcol],
                y=self.fit_results['input_df'][self.ycol],
                c='none',
                s=40,
                alpha=unbinned_alpha,
                marker='o',
                edgecolors='#607D8B',
                label=f"{self.ycol} ({len(self.fit_results['input_df'][self.ycol])} values)")

        # Binned data
        if self.usebins > 0:
            line_xy = ax.scatter(
                x=self.fit_results['bin_df'][self.xcol]['mean'],
                y=self.fit_results['bin_df'][self.ycol]['mean'],
                edgecolor=bin_edgecolor, color=bin_color, alpha=bin_alpha, s=bin_size,
                label=_label,
                zorder=1, marker=bin_marker)

        if show_bin_variation and (self.usebins > 0):
            if self.bins_y_agg == 'mean':
                ax.errorbar(
                    x=self.fit_results['bin_df'][self.xcol][self.bins_y_agg],
                    y=self.fit_results['bin_df'][self.ycol][self.bins_y_agg],
                    xerr=self.fit_results['bin_df'][self.xcol]['std'],
                    yerr=self.fit_results['bin_df'][self.ycol]['std'],
                    elinewidth=5, ecolor='black', alpha=.3, lw=0,
                    zorder=2, label="mean bins SD")

            elif self.bins_y_agg == 'median':
                ax.fill_between(
                    self.fit_results['bin_df'][self.xcol][self.bins_y_agg],
                    self.fit_results['bin_df'][self.ycol]['q25'],
                    self.fit_results['bin_df'][self.ycol]['q75'],
                    alpha=.3, zorder=2, color='black',
                    label="median bins IQR")

        # Highlight year
        if highlight_year:
            _subset_year = pd.DataFrame()
            _subset_year['bins_x'] = self.fit_results['bins_x']
            _subset_year['bins_y'] = self.fit_results['bins_y']
            _subset_year.index = pd.to_datetime(_subset_year.index)
            _subset_year = _subset_year.loc[_subset_year.index.year == highlight_year, :]
            _numvals_y = len(_subset_year['bins_y'])
            line_highlight = ax.scatter(
                _subset_year['bins_x'],
                _subset_year['bins_y'],
                edgecolor='#455A64', color='#FFD54F',  # amber 300
                alpha=1, s=100,
                label=f"{label} {highlight_year} ({_numvals_y} days)",
                zorder=98, marker=marker)
        else:
            line_highlight = None

        # Fit
        line_fit = line_fit_ci = None
        line_fit_pb = None
        if showfit:

            # rf"$y = {a:.4f}x{operator1}{b:.4f}$"

            _label_line_fit = f"${self.fit_results['fit_equation_str']}, r^2={self.fit_results['fit_r2']:.3f}$"
            line_fit, = ax.plot(self.fit_results['fit_df']['fit_x'],
                                self.fit_results['fit_df']['nom'],
                                c=fitline_color, lw=3, zorder=3, alpha=1, label=_label_line_fit)

            # Fit confidence region
            # Uncertainty lines (95% confidence)
            line_fit_ci = ax.fill_between(self.fit_results['fit_df']['fit_x'],
                                          self.fit_results['fit_df']['nom_lower_ci95'],
                                          self.fit_results['fit_df']['nom_upper_ci95'],
                                          alpha=.2, color=fitline_color, zorder=2,
                                          label="95% confidence region")

            # Fit prediction interval
            if show_prediction_interval:
                # Lower prediction band (95% confidence)
                ax.plot(self.fit_results['fit_df']['fit_x'],
                        self.fit_results['fit_df']['lower_predband'],
                        color=fitline_color, ls='--', zorder=3, lw=2,
                        label="95% prediction interval")
                # Upper prediction band (95% confidence)
                line_fit_pb, = ax.plot(self.fit_results['fit_df']['fit_x'],
                                       self.fit_results['fit_df']['upper_predband'],
                                       color=fitline_color, ls='--', zorder=3, lw=2,
                                       label=None)
        # Format
        if not title:
            title = (f"{self.bins_y_agg} {self.ycol} "
                     f"in {len(self.fit_results['bin_df'])} bins of {self.xcol}")
        else:
            pass
        ax.set_title(title, fontsize=16)
        _xlabel = xlabel if xlabel else self.fit_results['xvar']
        _ylabel = ylabel if ylabel else self.fit_results['yvar']
        default_format(ax=ax,
                       ax_xlabel_txt=_xlabel,
                       ax_ylabel_txt=_ylabel,
                       showgrid=True)
        add_zeroline_y(ax=ax, data=self.fit_results['input_df'][self.ycol])
        default_legend(ax=ax, ncol=1, textsize=13)
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        # ax.locator_params(axis='x', nbins=self.usebins)
        # ax.locator_params(axis='y', nbins=self.usebins)
        # default_legend(ax=ax, ncol=1, loc=(.07, .75))
        fig.show()
        return line_xy, line_fit, line_fit_ci, line_fit_pb, line_highlight


def example():
    from diive.configs.exampledata import load_exampledata_parquet

    df_orig = load_exampledata_parquet()
    # df_orig = df_orig.loc[df_orig.index.year == 2021].copy()

    # Variables
    vpd_col = 'VPD_f'
    ta_col = 'Tair_f'
    nee_col = 'NEE_CUT_REF_f'
    xcol = vpd_col
    ycol = nee_col

    subset = df_orig.loc[(df_orig.index.month >= 6) & (df_orig.index.month <= 9)].copy()
    subset = subset.loc[subset['Rg_f'] > 200]

    # Convert units
    subset[vpd_col] = subset[vpd_col].multiply(0.1)  # hPa --> kPa
    subset[nee_col] = subset[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    # y_units = "kPa"
    y_units = "gCO_{2}\ m^{-2}\ 30min^{-1}"
    ylabel = f"NEE (${y_units}$)"
    # ylabel = f"VPD (${y_units}$)"
    x_units = "°C"
    xlabel = f"air temperature (${x_units}$)"

    bf = BinFitterCP(
        df=subset,
        # n_bootstraps=2,
        xcol=xcol,
        ycol=ycol,
        # predict_max_x=None,
        # predict_min_x=None,
        n_predictions=1000,
        n_bins_x=0,
        bins_y_agg='mean',
        fit_type='quadratic_offset'  # 'linear', 'quadratic_offset', 'quadratic', 'cubic'
    )
    bf.run()
    fit_results = bf.fit_results
    # bf.showplot_binfitter(highlight_year=None, xlabel=xlabel, ylabel=ylabel)

    bf.showplot(
        show_unbinned_data=True,
        show_bin_variation=False,
        showfit=True,
        title='',
        xlabel=xlabel,
        ylabel=ylabel,
        unbinned_alpha=.5
        # xlim=(0, 30),
        # ylim=(-1, 0)
    )


if __name__ == '__main__':
    example()
