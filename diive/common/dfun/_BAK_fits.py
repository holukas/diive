"""
DATA FUNCTIONS: FITS
====================

# last update in: v0.23.0

This module is part of DIIVE:
https://gitlab.ethz.ch/holukas/diive

"""

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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import stats
from scipy.optimize import curve_fit

from pkgs.dfun.stats import q25, q75

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


def groupagg(df, num_bins, bin_col):
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
    numvals_in_group = \
        grouped_df[(bin_col)]['count'].describe()[['min', 'max']]

    # print(numvals_in_group)

    # Bins info
    grouped_df['BIN_START'] = bins[0:-1]

    return grouped_df


class LineCrossingBTS:
    """"""

    def __init__(
            self,
            df: pd.DataFrame,
            x_col: str or tuple,
            y1_col: str or tuple,
            y2_col: str or tuple,
            ignore_zero_x: bool = False,
            num_bts_runs: int = 10,
            num_groups: int = 40
    ):
        self.df = df[[x_col, y1_col, y2_col]].dropna()  # Remove NaNs
        self.x_col = x_col
        self.ignore_zero_x = ignore_zero_x
        self.y1_col = y1_col
        self.y2_col = y2_col
        self.num_bts_runs = num_bts_runs
        self.num_groups = num_groups

        # Collect results from different bootstrap runs
        self.bts = None
        self.bts_line_crossings_x = []
        self.bts_line_crossings_y1 = []
        self.bts_line_crossings_y2 = []

    def get_linecrossings(self):
        """Return results from bootstrap runs"""
        return self.bts_line_crossings_x, self.bts_line_crossings_y1, self.bts_line_crossings_y2

    def check_linecrossings(self):
        """Print results to screen"""

        lc_x = self.bts_line_crossings_x
        lc_y1 = self.bts_line_crossings_y1
        lc_y2 = self.bts_line_crossings_y2

        print(
            f"Line crossings after {self.bts + 1} bootstrap runs\n"
            f"Line crossings x: {lc_x}\n"
            f"Line crossings y1: {lc_y1}\n"
            f"Line crossings y2: {lc_y2}\n"

            f"Line crossings x (0.05 / 0.5 / 0.95): "
            f"{np.quantile(lc_x, 0.05):.3f}"
            f" / {np.median(lc_x):.3f}"
            f" / {np.quantile(lc_x, 0.95):.3f}\n"

            f"Line crossings y1 (0.05 / 0.5 / 0.95): "
            f"{np.quantile(lc_y1, 0.05):.3f}"
            f" / {np.median(lc_y1):.3f}"
            f" / {np.quantile(lc_y1, 0.95):.3f}\n"

            f"Line crossings y2 (0.05 / 0.5 / 0.95): "
            f"{np.quantile(lc_y2, 0.05):.3f}"
            f" / {np.median(lc_y2):.3f}"
            f" / {np.quantile(lc_y2, 0.95):.3f}\n"
        )

    def detect_linecrossings(self):

        # Bootstrap for line crossing
        for self.bts in range(0, self.num_bts_runs):
            print(f"Bootstrap run #{self.bts}")

            # Sample from data
            sample_df = self.df.sample(n=int(len(self.df)), replace=True)

            if self.ignore_zero_x:
                sample_df = sample_df.loc[sample_df[self.x_col] > 0]

            # sample_df = groupagg(df=sample_df,
            #                      grouping_col=self.x_col,
            #                      num_groups=self.num_groups)

            # Divide into groups of x
            group, bins = pd.qcut(sample_df[self.x_col],
                                  q=self.num_groups,
                                  retbins=True)  # How awesome!
            sample_df['group'] = group

            sample_df.sort_index(axis=1, inplace=True)  # lexsort for better performance

            # Calc stats for each group
            grouped_df = \
                sample_df.groupby('group').agg(
                    {'mean', 'median', 'count', 'std', q25, q75})
            numvals_in_group = \
                grouped_df[(self.x_col)]['count'].describe()[['min', 'max']]

            # Bins info
            grouped_df[self.bin_start_col] = bins[0:-1]

            try:
                # y1
                fitter = FitterCP(x=grouped_df[self.bin_start_col],
                                  y=grouped_df[self.y1_col]['median'],
                                  predict_max_x=None,
                                  predict_min_x=None,
                                  num_predictions=10000)
                y1_fit_df, y1_fit_params = fitter._fit()

                # y2
                fitter = FitterCP(x=grouped_df[self.bin_start_col],
                                  y=grouped_df[self.y2_col]['median'],
                                  predict_max_x=None,
                                  predict_min_x=None,
                                  num_predictions=10000)
                y2_fit_df, y2_fit_params = fitter._fit()

                crossing_df = pd.DataFrame()
                crossing_df['nom_ix'] = y1_fit_df['fit_x']
                crossing_df['nom_y1'] = y1_fit_df['nom']
                crossing_df['nom_y2'] = y2_fit_df['nom']
                # crossing_df = crossing_df.set_index('nom_ix')

                # Find where the two fits cross
                crossing_df['diff'] = crossing_df['nom_y1'].sub(crossing_df['nom_y2'])
                cross_ix = np.argmax(crossing_df['diff'] < 0)
                if cross_ix == 0:
                    cross_ix = np.argmax(crossing_df['diff'] > 0)

                if cross_ix != 0:
                    linecross_x = crossing_df.iloc[cross_ix]['nom_ix']
                    linecross_y1 = crossing_df.iloc[cross_ix]['nom_y1']
                    linecross_y2 = crossing_df.iloc[cross_ix]['nom_y2']

                    self.bts_line_crossings_x.append(linecross_x)
                    self.bts_line_crossings_y1.append(linecross_y1)
                    self.bts_line_crossings_y2.append(linecross_y2)

            except:
                # Sometimes the fit to the bootstrapped data fails, in that case repeat
                self.bts -= 1


class FitterCP:
    """Fit function to data and give CI and bootstrapped PI"""

    def __init__(
            self,
            df: pd.DataFrame,
            x_col: str or tuple,
            y_col: str or tuple,
            predict_max_x: float = None,
            predict_min_x: float = None,
            num_predictions: int = None,
            bins_x_num: int = 0,
            bins_y_agg: str = None,
    ):
        self.df = df[[x_col, y_col]].dropna()  # Remove NaNs, working data
        self.x_col = x_col
        self.y_col = y_col
        self.x = self.df[self.x_col]
        self.y = self.df[self.y_col]
        self.len_y = len(self.y)
        self.num_predictions = num_predictions
        self.usebins = bins_x_num if bins_x_num >= 0 else 0  # Must be positive
        self.bins_y_agg = bins_y_agg
        self.fit_x_max = predict_max_x if isinstance(predict_max_x, float) else self.x.max()
        self.fit_x_min = predict_min_x if isinstance(predict_min_x, float) else self.x.min()
        self.num_predictions = num_predictions if isinstance(num_predictions, int) else len(self.x)
        self.num_predictions = 2 if self.num_predictions < 2 else self.num_predictions

        self.fit_results = {}  # Stores fit results
        # self.bts_results = {}  # Stores fit results for each bootstrap run
        # self.bts_predbands_minmax = pd.DataFrame()  # Min/max for prediction bands across bootstraps

    def run(self):

        bts_upper_predbands_df = pd.DataFrame()
        bts_lower_predbands_df = pd.DataFrame()

        # Sample from original input data and collect results
        if self.bootstrap_runs > 0:
            for bts in range(1, self.bootstrap_runs + 1):
                print(f"Fitting {self.y_col}, bootstrap run #{bts}")
                try:
                    bts_df = self.df.sample(n=int(len(self.df)), replace=True)
                    self.bts_results[bts] = self._fit(df=bts_df)
                    # Collect prediction bands (and their x values)
                    bts_upper_predbands_df[bts] = self.bts_results[bts]['fit_df']['upper_predband']
                    bts_lower_predbands_df[bts] = self.bts_results[bts]['fit_df']['lower_predband']
                except:
                    print(f"Fitting failed, repeating run {bts}")
                    bts -= 1

            # fit_x is the same across bootstraps, is set at start of FitterCP
            self.bts_predbands_minmax['fit_x'] = self.bts_results[1]['fit_df']['fit_x']

            # For each predicted value, calculate quantiles
            self.bts_predbands_minmax['upper_predband_bts_max'] = bts_upper_predbands_df.T.quantile(0.975)
            self.bts_predbands_minmax['upper_predband_bts_min'] = bts_upper_predbands_df.T.quantile(0.025)
            self.bts_predbands_minmax['lower_predband_bts_max'] = bts_lower_predbands_df.T.quantile(0.975)
            self.bts_predbands_minmax['lower_predband_bts_min'] = bts_lower_predbands_df.T.quantile(0.025)

        # Original input data
        self.fit_results = self._fit(df=self.df.copy())

    def plot_results(self):
        fig = plt.figure(figsize=(9, 12))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.1, right=0.9, top=0.97, bottom=0.03)

        # Results from original data
        ax1 = fig.add_subplot(gs[0, 0])
        self._make_plot(ax=ax1, results=self.fit_results, predbands_df=self.bts_predbands_minmax)

        # Results from bootstrapped data
        ax2 = fig.add_subplot(gs[1, 0])
        for bts_run in self.bts_results.keys():
            self._make_plot(ax=ax2, results=self.bts_results[bts_run])

        fig.show()

    def _make_plot(self, ax, results, predbands_df: pd.DataFrame = None):
        # Get data
        x = results['x']
        y = results['y']
        fit_df = results['fit_df']

        # Daily aggregates
        ax.scatter(x, y, edgecolor='#B0BEC5', color='none',
                   alpha=.5, s=60, label='daily aggregates', zorder=9, marker='o')

        # Fit + CI
        label_fit = r"$y = ax^2 + bx + c$"
        ax.plot(fit_df['fit_x'], fit_df['nom'], lw=2, color='black',
                label=label_fit, zorder=11)
        ax.fill_between(fit_df['fit_x'],
                        fit_df['nom_lower_ci95'],
                        fit_df['nom_upper_ci95'], alpha=.2, zorder=10,
                        color='black',
                        label="95% confidence region")  # uncertainty lines (95% confidence)

        # Prediction bands
        # # Lower prediction band (95% confidence)
        ax.plot(fit_df['fit_x'], fit_df['lower_predband'], color='black',
                ls='--', zorder=8,
                label="95% prediction interval")
        # # Upper prediction band (95% confidence)
        ax.plot(fit_df['fit_x'], fit_df['upper_predband'], color='black',
                ls='--', zorder=8)

        if isinstance(predbands_df, pd.DataFrame):
            ax.fill_between(predbands_df['fit_x'],
                            predbands_df['upper_predband_bts_max'],
                            predbands_df['upper_predband_bts_min'], alpha=.1, zorder=8,
                            color='black', label="95% prediction interval")
            ax.fill_between(predbands_df['fit_x'],
                            predbands_df['lower_predband_bts_max'],
                            predbands_df['lower_predband_bts_min'], alpha=.1, zorder=8,
                            color='black', label="95% prediction interval")

        ax.axhline(0, lw=1, color='black', zorder=12)

    def get_results(self):
        return self.fit_results, self.bts_results, self.bts_predbands_minmax

    def _bin_data(self, df, num_bins: int = 10):
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

    def _func(self, x, a, b, c):
        """Fitting function"""
        return a * x ** 2 + b * x + c

    def _fit(self, df):
        """Calculate curve fit, confidence intervals and prediction bands

        kudos: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
        """

        # Bin data
        if self.usebins == 0:
            x = df[self.x_col]
            y = df[self.y_col]
            len_y = len(y)
        else:
            df = self._bin_data(df=df, num_bins=self.usebins)
            x = df['BIN_START']
            y = df[self.y_col][self.bins_y_agg]
            len_y = len(self.y)

        # Fit function f to data
        fit_params_opt, fit_params_cov = curve_fit(self._func, x, y)

        # Retrieve parameter values
        a = fit_params_opt[0]
        b = fit_params_opt[1]
        c = fit_params_opt[2]

        # Calc r2
        fit_r2 = \
            1.0 - (sum((y - self._func(x, a, b, c)) ** 2) / ((len_y - 1.0) * np.var(y, ddof=1)))

        # Calculate parameter confidence interval
        a, b, c = unc.correlated_values(fit_params_opt, fit_params_cov)

        # Calculate regression confidence interval
        fit_x = np.linspace(self.fit_x_min, self.fit_x_max, self.num_predictions)
        fit_y = a * fit_x ** 2 + b * fit_x + c
        nom = unp.nominal_values(fit_y)
        std = unp.std_devs(fit_y)

        # Best lower and upper prediction bands
        lower_predband, upper_predband = \
            self._predband(px=fit_x, x=x, y=y,
                           params_opt=fit_params_opt, func=self._func, conf=0.95)

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
                           yvar=self.y_col)

        return fit_results


if __name__ == '__main__':
    pass


def fit_to_bins_linreg(df, x_col, y_col, bin_col):
    # https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
    # https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

    from sklearn import linear_model
    from sklearn import metrics
    import numpy as np

    _df = df.copy()

    X = _df[x_col][bin_col].values.reshape(-1, 1)  # Attributes (independent)
    y = _df[y_col][bin_col].values.reshape(-1, 1)  # Labels (dependent, predicted)

    # Fitting the linear Regression model on two components.
    linreg = linear_model.LinearRegression()
    linreg.fit(X, y)  # training the algorithm
    y_pred = linreg.predict(X)  # predict y
    predicted_col = (y_col[0], y_col[1], 'predicted')
    _df[predicted_col] = y_pred.flatten()

    # # Robustly fit linear model with RANSAC algorithm
    # # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py
    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(X, y)
    # inlier_mask = ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

    predicted_results = {
        'MAE': metrics.mean_absolute_error(y, y_pred),
        'MSE': metrics.mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y, y_pred)),
        'r2': metrics.explained_variance_score(y, y_pred),
        'intercept': float(linreg.intercept_),
        'slope': float(linreg.coef_)
    }

    # predicted_score = regressor.score(X, y)  # retrieve r2
    # predicted_intercept = regressor.intercept_  # retrieve the intercept
    # predicted_slope = regressor.coef_  # retrieving the slope

    # comparison_df = pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()})

    return _df, predicted_col, predicted_results

    # # split 80% of the data to the training set while 20% of the data to test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # # Fitting the linear Regression model on two components.
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)  # training the algorithm
    # y_pred = regressor.predict(X_test)
    # comparison_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    # lin_df['predicted'] = lin.predict(X)
    #
    # plt.plot(X, lin.predict(X), color='red')
    # x_bin_avg_df['x'] = lin.predict(X)
    #
    # # Fitting Polynomial Regression to the dataset
    # from sklearn.preprocessing import PolynomialFeatures
    #
    # poly = PolynomialFeatures(degree=4)
    # X_poly = poly.fit_transform(X)
    #
    # poly.fit(X_poly, y)
    # lin2 = LinearRegression()
    # lin2.fit(X_poly, y)


def fit_to_bins_polyreg(df, x_col, y_col, bin_col, degree):
    # https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
    # https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions

    # * https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
    # https://stackoverflow.com/questions/39012611/how-get-equation-after-fitting-in-scikit-learn
    # https://stackoverflow.com/questions/51006193/interpreting-logistic-regression-feature-coefficient-values-in-sklearn

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import numpy as np

    _df = df.copy()

    X = _df[x_col][bin_col].values.reshape(-1, 1)  # Attributes (independent)
    y = _df[y_col][bin_col].values.reshape(-1, 1)  # Labels (dependent, predicted)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin = LinearRegression()
    lin.fit(X_poly, y)

    y_pred = lin.predict(poly.fit_transform(X))

    predicted_col = (y_col[0], y_col[1], 'predicted')
    _df[predicted_col] = y_pred.flatten()

    predicted_results = {
        'MAE': metrics.mean_absolute_error(y, y_pred),
        'MSE': metrics.mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y, y_pred)),
        'r2': metrics.explained_variance_score(y, y_pred),
        'intercept': float(lin.intercept_),
        'slope': lin.coef_}

    # The logistic function, which returns the probability of success,
    # is given by p(x) = 1/(1 + exp(-(B0 + B1X1 + ... BnXn)). B0 is in intercept.
    # B1 through Bn are the coefficients. X1 through Xn are the features.

    return _df, predicted_col, predicted_results
