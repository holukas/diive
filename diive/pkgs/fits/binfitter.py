"""
Fit with CI and PI

CI ... confidence interval
PI ... prediction interval

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
- https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit
- https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
- https://towardsdatascience.com/calculating-confidence-interval-with-bootstrapping-872c657c058d
- https://lmfit.github.io/lmfit-py/
- **https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics**
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

"""
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame
from scipy import stats
from scipy.optimize import curve_fit

from diive.core.dfun.stats import q25, q75
from diive.core.plotting.plotfuncs import save_fig, default_format, add_zeroline_y
from diive.core.plotting.styles.LightTheme import FONTSIZE_LEGEND


def groupagg(df, num_bins, bin_col) -> DataFrame:
    # Divide into groups of x

    # # Alternative: using .cut
    # group, bins = pd.cut(df[bin_col],
    #                       bins=num_bins,
    #                       retbins=True,
    #                       duplicates='raise')  # How awesome!

    # Alternative: using .qcut
    group, bins = pd.qcut(df[bin_col],
                          q=num_bins,
                          retbins=True,
                          duplicates='drop')  # How awesome!

    df['group'] = group

    df.sort_index(axis=1, inplace=True)  # lexsort for better performance

    # Calc stats for each group
    grouped_df = \
        df.groupby('group').agg(
            {'mean', 'median', 'max', 'min', 'count', 'std', q25, q75})

    # print(numvals_in_group)

    # Bins info
    grouped_df['BIN_START'] = bins[0:-1]

    return grouped_df


class BinFitterBTS:
    """Use bootstrapped data in multiple BinFitter runs"""

    def __init__(self,
                 df: DataFrame,
                 n_bootstraps: int = 10,
                 **params):
        self.df = df.copy()
        self.n_bootstraps = n_bootstraps
        self.params = params

        self._fit_results = {}
        self._bts_fit_results = {}

    @property
    def fit_results(self) -> dict:
        """Return fit results"""
        if self._fit_results is None:
            raise Exception('No fit results available.')
        return self._fit_results

    @property
    def bts_fit_results(self) -> dict:
        """Return bootstrapped fit results"""
        if not self._bts_fit_results:
            raise Exception('No fit results available.')
        return self._bts_fit_results

    def fit(self):

        # Non-bootstrapped fit results
        bf = BinFitter(df=self.df, **self.params)
        bf.fit()
        self._fit_results = bf.fit_results

        n_bts_successful = 0  # Number of succesful bootstrapping runs

        while n_bts_successful < self.n_bootstraps:
            try:
                bts_df = self.df.sample(n=int(len(self.df)), replace=True, random_state=None)  # Bootstrap data
                bts_df = bts_df.sort_index()
                bf = BinFitter(df=bts_df, **self.params)
                bf.fit()
                self._bts_fit_results[n_bts_successful] = bf.fit_results  # Store bootstrapped results in dict
                print(f"Bootstrapping run {n_bts_successful} for fit line successful ... ")
                n_bts_successful += 1
            except:
                print(f"Bootstrapping for fit line failed, repeating run {n_bts_successful} ... ")
                pass

        self._collect_predband_quantiles()

    def _collect_predband_quantiles(self):
        # Collect bootstrapping results
        _fit_x_predbands = pd.DataFrame()
        _upper_predbands = pd.DataFrame()
        _lower_predbands = pd.DataFrame()
        for bts_run in range(0, self.n_bootstraps):
            _fit_x_predbands[bts_run] = self.bts_fit_results[bts_run]['fit_df']['fit_x'].copy()
            _upper_predbands[bts_run] = self.bts_fit_results[bts_run]['fit_df']['upper_predband'].copy()
            _lower_predbands[bts_run] = self.bts_fit_results[bts_run]['fit_df']['lower_predband'].copy()
        self._fit_results['fit_df']['bts_predband_fit_x'] = _fit_x_predbands.mean(
            axis=1)  # Output is the same for all bootstraps
        self._fit_results['fit_df']['bts_upper_predband_Q97.5'] = _upper_predbands.quantile(q=.975, axis=1)
        self._fit_results['fit_df']['bts_upper_predband_Q02.5'] = _upper_predbands.quantile(q=.025, axis=1)
        self._fit_results['fit_df']['bts_lower_predband_Q97.5'] = _lower_predbands.quantile(q=.975, axis=1)
        self._fit_results['fit_df']['bts_lower_predband_Q02.5'] = _lower_predbands.quantile(q=.025, axis=1)

    def showplot_binfitter(self,
                           saveplot: bool = False,
                           title: str = None,
                           path: Path or str = None,
                           **kwargs):
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        # ax = self.plot_binfitter(ax=ax, **kwargs)
        PlotBinFitterBTS(ax=ax, fit_results=self.fit_results, **kwargs).plot_binfitter()  # todo
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)


class PlotBinFitterBTS:

    def __init__(self,
                 fit_results: dict,
                 ax,
                 xlabel: str = None,
                 ylabel: str = None,
                 highlight_year: int = None,
                 color: str = 'none',
                 color_fitline: str = '#4CAF50',
                 label: str = 'label',
                 edgecolor: str = '#B0BEC5',
                 marker: str = 'o',
                 alpha: float = .9,
                 showfit: bool = True,
                 show_prediction_interval: bool = True,
                 size_scatter: int = 60,
                 fit_type: str = 'quadratic'
                 ):
        self.fit_results = fit_results
        self.ax = ax
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.highlight_year = highlight_year
        self.color = color
        self.color_fitline = color_fitline
        self.label = label
        self.edgecolor = edgecolor
        self.marker = marker
        self.alpha = alpha
        self.showfit = showfit
        self.show_prediction_interval = show_prediction_interval
        self.size_scatter = size_scatter
        self.fit_type = fit_type

    def plot_binfitter(self):
        # Data
        xdata = self.fit_results['x']
        ydata = self.fit_results['y']

        # Plot data
        _label = self._plot_setlabel(label=self.label)
        line_xy = self.ax.scatter(xdata, ydata,
                                  edgecolor=self.edgecolor, color=self.color, alpha=self.alpha,
                                  s=self.size_scatter, label=_label, zorder=1, marker=self.marker)

        # Highlight year
        line_highlight = None
        if self.highlight_year:
            line_highlight = self._plot_highlight_year(ax=self.ax, highlight_year=self.highlight_year,
                                                       marker=self.marker,
                                                       label=self.label)

        # Fit and confidence intervals
        line_fit = line_fit_ci = None
        line_fit_pb = None
        if self.showfit:
            line_fit = self._plot_fit(ax=self.ax, fit_type=self.fit_type, color_fitline=self.color_fitline)
            line_fit_ci = self._plot_fit_ci(ax=self.ax, color_fitline=self.color_fitline)

        # Prediction bands
        if self.show_prediction_interval:
            line_fit_pb = self._plot_fit_pi(ax=self.ax, color_fitline=self.color_fitline)

        # Format
        default_format(ax=self.ax, txt_xlabel=self.xlabel, txt_ylabel=self.ylabel)
        add_zeroline_y(ax=self.ax, data=ydata)

        # return line_xy, line_fit, line_fit_ci, line_fit_pb, line_highlight

        # def _plot_custom_legend(self, ax, ):
        # Custom legend
        # Assign two of the handles to the same legend entry by putting them in a tuple
        # and using a generic handler map (which would be used for any additional
        # tuples of handles like (p1, p3)).
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
        l = self.ax.legend(
            [
                # line_crd_vertical if showline_crd else None,
                # range_bts_netzeroflux if showrange_crd else None,
                # line_xcrd_vertical if showline_xcrd else None,
                line_xy,
                line_highlight if line_highlight else None,
                line_fit if self.showfit else None,
                line_fit_ci if self.showfit else None,
                line_fit_pb if self.show_prediction_interval else None
            ],
            [
                # line_crd_vertical.get_label() if showline_crd else None,
                # range_bts_netzeroflux.get_label() if showrange_crd else None,
                # line_xcrd_vertical.get_label() if showline_xcrd else None,
                line_xy.get_label(),
                line_highlight.get_label() if line_highlight else None,
                line_fit.get_label() if self.showfit else None,
                line_fit_ci.get_label() if self.showfit else None,
                line_fit_pb.get_label() if self.show_prediction_interval else None
            ],
            bbox_to_anchor=(0, 1),
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
            loc="lower left",
            ncol=2,
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)})

        return self.ax, line_xy,line_highlight,line_fit,line_fit_ci,line_fit_pb

    def _plot_highlight_year(self, ax, highlight_year, label, marker):
        _subset_year = pd.DataFrame()
        _subset_year['x'] = self.fit_results['x']
        _subset_year['y'] = self.fit_results['y']
        _subset_year.index = pd.to_datetime(_subset_year.index)
        _subset_year = _subset_year.loc[_subset_year.index.year == highlight_year, :]
        _numvals_y = len(_subset_year['y'])
        line_highlight = ax.scatter(_subset_year['x'],
                                    _subset_year['y'],
                                    edgecolor='#455A64', color='#FFD54F',  # amber 300
                                    alpha=1, s=100,
                                    label=f"{label} {highlight_year} ({_numvals_y} values)",
                                    zorder=98, marker=marker)
        return line_highlight

    def _plot_setlabel(self, label):
        # x/y
        _numvals_y = len(self.fit_results['y'])
        try:
            # Add year info to label, if available
            _startyear_y = self.fit_results['y'].index.year[0]
            _endyear_y = self.fit_results['y'].index.year[-1]
            label = f"{label} {_startyear_y}-{_endyear_y} ({_numvals_y} values)"
        except AttributeError:
            label = label
        return label

    def _plot_fit_ci(self, ax, color_fitline):
        # Fit confidence region
        # Uncertainty lines (95% confidence)
        line_fit_ci = ax.fill_between(self.fit_results['fit_df']['fit_x'],
                                      self.fit_results['fit_df']['nom_lower_ci95'],
                                      self.fit_results['fit_df']['nom_upper_ci95'],
                                      alpha=.2, color=color_fitline, zorder=97,
                                      label="95% confidence region")
        return line_fit_ci

    def _plot_fit_pi(self, ax, color_fitline):
        # Fit prediction interval
        # Lower prediction band (95% confidence)
        ax.plot(self.fit_results['fit_df']['fit_x'],
                self.fit_results['fit_df']['lower_predband'],
                color=color_fitline, ls='--', zorder=97, lw=2,
                label="95% prediction interval")
        ax.fill_between(self.fit_results['fit_df']['fit_x'],
                                      self.fit_results['fit_df']['bts_lower_predband_Q97.5'],
                                      self.fit_results['fit_df']['bts_lower_predband_Q02.5'],
                                      alpha=.2, color=color_fitline, zorder=97,
                                      label="95% confidence region")
        # Upper prediction band (95% confidence)
        line_fit_pb, = ax.plot(self.fit_results['fit_df']['fit_x'],
                               self.fit_results['fit_df']['upper_predband'],
                               color=color_fitline, ls='--', zorder=96, lw=2,
                               label="95% prediction interval")
        ax.fill_between(self.fit_results['fit_df']['fit_x'],
                                      self.fit_results['fit_df']['bts_upper_predband_Q97.5'],
                                      self.fit_results['fit_df']['bts_upper_predband_Q02.5'],
                                      alpha=.2, color=color_fitline, zorder=97,
                                      label="95% confidence region")
        return line_fit_pb

    def _plot_fit(self, ax, fit_type, color_fitline):
        a = self.fit_results['fit_params_opt'][0]
        b = self.fit_results['fit_params_opt'][1]
        operator1 = "+" if b > 0 else ""
        label_fit = None
        if fit_type == 'linear':
            label_fit = rf"$y = {a:.4f}x{operator1}{b:.4f}$"
        elif fit_type == 'quadratic':
            c = self.fit_results['fit_params_opt'][2]
            operator2 = "+" if c > 0 else ""
            label_fit = rf"$y = {a:.4f}x^2{operator1}{b:.4f}x{operator2}{c:.4f}$"
        line_fit, = ax.plot(self.fit_results['fit_df']['fit_x'],
                            self.fit_results['fit_df']['nom'],
                            c=color_fitline, lw=3, zorder=99, alpha=1, label=label_fit)
        return line_fit


class BinFitter:
    """Fit function to (binned) data and give CI and bootstrapped PI"""

    def __init__(self,
                 df: pd.DataFrame,
                 x_col: str or tuple,
                 y_col: str or tuple,
                 predict_max_x: float = None,
                 predict_min_x: float = None,
                 num_predictions: int = None,
                 bins_x_num: int = 0,
                 bins_y_agg: str = None,
                 fit_type: Literal['linear', 'quadratic'] = 'quadratic'
                 ):
        self.df = df[[x_col, y_col]].copy().dropna()  # Remove NaNs, working data
        self.x_col = x_col
        self.y_col = y_col
        self.x = self.df[self.x_col]
        self.y = self.df[self.y_col]
        self.bins_y_agg = bins_y_agg
        self.num_predictions = num_predictions
        self.usebins = bins_x_num if bins_x_num >= 0 else 0  # Must be positive
        self.fit_x_max = predict_max_x if isinstance(predict_max_x, float) else self.x.max()
        self.fit_x_min = predict_min_x if isinstance(predict_min_x, float) else self.x.min()
        self.num_predictions = num_predictions if isinstance(num_predictions, int) else len(self.x)
        self.num_predictions = 2 if self.num_predictions < 2 else self.num_predictions

        self.equation = self._set_fit_equation(type=fit_type)
        self.fit_type = fit_type

        self._fit_results = {}  # Stores fit results

    @property
    def fit_results(self) -> dict:
        if not self._fit_results:
            raise Exception('Fit results not available.')
        return self._fit_results

    def fit(self):
        self._fit_results = self._fit(df=self.df.copy())

    def get_results(self):
        return self._fit_results

    def _bin_data(self, df, num_bins: int = 10) -> pd.DataFrame:
        return groupagg(df=df, num_bins=num_bins, bin_col=self.x_col)

    def _predband(self, px, x, y, params_opt, func, conf=0.95):
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

    def _set_fit_equation(self, type: str = 'quadratic'):
        if type == 'quadratic':
            equation = self._fit_quadratic
        elif type == 'linear':
            equation = self._fit_linear
        else:
            equation = self._fit_quadratic
        return equation

    def _fit_linear(self, x, a, b):
        """Linear fit"""
        return a * x + b

    def _fit_quadratic(self, x, a, b, c):
        """Quadratic equation"""
        return a * x ** 2 + b * x + c

    # def _func(self, x, a, b, c):
    #     """Fitting function"""
    #     return a * x ** 2 + b * x + c

    def _set_fit_data(self, df):
        # Bin data
        numvals_per_bin = {}
        if self.usebins == 0:
            _df = df.copy()
            x = self.x
            y = self.y
            len_y = len(y)
            numvals_per_bin['min'] = len_y
            numvals_per_bin['max'] = len_y
        else:
            _df = self._bin_data(df=df, num_bins=self.usebins)
            x = _df['BIN_START']
            y = _df[self.y_col][self.bins_y_agg]
            len_y = len(self.y)
            numvals_per_bin = \
                _df[self.y_col]['count'].describe()[['min', 'max']].to_dict()
        return _df, x, y, len_y, numvals_per_bin

    def _fit(self, df):
        """Calculate curve fit, confidence intervals and prediction bands

        kudos: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
        """

        df, x, y, len_y, numvals_per_bin = self._set_fit_data(df=df)

        # Fit function f to data
        fit_params_opt, fit_params_cov = curve_fit(self.equation, x, y)

        # Retrieve parameter values
        a = fit_params_opt[0]
        b = fit_params_opt[1]
        c = fit_params_opt[2] if self.fit_type == 'quadratic' else None

        # Calc r2
        kwargs = None
        if self.fit_type == 'quadratic':
            kwargs = dict(x=x, a=a, b=b, c=c)
        elif self.fit_type == 'linear':
            kwargs = dict(x=x, a=a, b=b)
        fit_r2 = 1.0 - (sum((y - self.equation(**kwargs)) ** 2) / ((len_y - 1.0) * np.var(y, ddof=1)))

        # Calculate parameter confidence interval
        c = None
        if self.fit_type == 'quadratic':
            a, b, c = unc.correlated_values(fit_params_opt, fit_params_cov)
        elif self.fit_type == 'linear':
            a, b = unc.correlated_values(fit_params_opt, fit_params_cov)

        # Calculate regression confidence interval
        fit_x = np.linspace(self.fit_x_min, self.fit_x_max, self.num_predictions)

        if self.fit_type == 'quadratic':
            fit_y = a * fit_x ** 2 + b * fit_x + c
        elif self.fit_type == 'linear':
            fit_y = a * fit_x + b

        nom = unp.nominal_values(fit_y)
        std = unp.std_devs(fit_y)

        # sample_df = self.df.sample(n=int(len(self.df)), replace=True)

        # Best lower and upper prediction bands
        lower_predband, upper_predband = \
            self._predband(px=fit_x, x=x, y=y,
                           params_opt=fit_params_opt, func=self.equation, conf=0.95)
        # lower_predband, upper_predband = \
        #     self._predband(px=fit_x, x=x, y=y,
        #                    params_opt=fit_params_opt, func=self._func, conf=0.95)

        # Fit data
        fit_df = pd.DataFrame()
        fit_df['fit_x'] = fit_x
        fit_df['fit_y'] = fit_y
        fit_df['std'] = std
        fit_df['nom'] = nom
        fit_df['lower_predband'] = lower_predband
        fit_df['upper_predband'] = upper_predband
        ## Calc 95% confidence region of fit
        fit_df['nom_lower_ci95'] = fit_df['nom'] - 1.96 * fit_df['std']
        fit_df['nom_upper_ci95'] = fit_df['nom'] + 1.96 * fit_df['std']

        # Collect results in dict
        fit_results = dict(fit_df=fit_df,
                           fit_params_opt=fit_params_opt,
                           fit_params_cov=fit_params_cov,
                           fit_r2=fit_r2,
                           x=x,
                           y=y,
                           xvar=self.x_col,
                           yvar=self.y_col,
                           fit_equation=self.equation,
                           numvals_per_bin=numvals_per_bin)

        return fit_results


def example():
    from diive.core.io.files import load_pickle

    # Variables
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    x_col = vpd_col
    y_col = nee_col

    # Load data, using pickle for fast loading
    source_file = r"L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\__apply\co2penalty_dav\input_data\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle"
    df_orig = load_pickle(filepath=source_file)
    df_orig = df_orig.loc[df_orig.index.year >= 2019].copy()

    # Select daytime data between May and September 1997-2021
    maysep_dt_df = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()
    maysep_dt_df = maysep_dt_df.loc[maysep_dt_df['PotRad_CUT_REF'] > 20]

    # Convert units
    maysep_dt_df[vpd_col] = maysep_dt_df[vpd_col].multiply(0.1)  # hPa --> kPa
    maysep_dt_df[nee_col] = maysep_dt_df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    x_units = "kPa"
    y_units = "gCO_{2}\ m^{-2}\ 30min^{-1}"
    xlabel = f"Half-hourly VPD ({x_units})"
    ylabel = f"{y_col} (${y_units}$)"

    bf = BinFitterBTS(
        df=maysep_dt_df,
        n_bootstraps=99,
        x_col=x_col,
        y_col=y_col,
        predict_max_x=maysep_dt_df[x_col].max(),
        predict_min_x=maysep_dt_df[x_col].min(),
        num_predictions=1000,
        bins_x_num=0,
        bins_y_agg=None,
        fit_type='quadratic'
    )
    bf.fit()
    fit_results = bf.fit_results
    bf.showplot_binfitter(highlight_year=2019, xlabel=xlabel, ylabel=ylabel)


if __name__ == '__main__':
    example()
