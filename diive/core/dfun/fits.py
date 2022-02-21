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
import numpy as np
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import stats
from scipy.optimize import curve_fit

# from gui import plotfuncs
from diive.core.dfun.stats import q25, q75

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


def groupagg(df, num_bins, bin_col) -> pd.DataFrame:
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


class BinFitterCP:
    """Fit function to (binned) data and give CI and bootstrapped PI"""

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
            fit_type: str = 'quadratic'
    ):
        self.df = df[[x_col, y_col]].dropna()  # Remove NaNs, working data
        self.x_col = x_col
        self.y_col = y_col
        self.x = self.df[self.x_col]
        self.y = self.df[self.y_col]
        self.len_y = len(self.y)
        self.bins_y_agg = bins_y_agg
        self.num_predictions = num_predictions
        self.usebins = bins_x_num if bins_x_num >= 0 else 0  # Must be positive
        self.fit_x_max = predict_max_x if isinstance(predict_max_x, float) else self.x.max()
        self.fit_x_min = predict_min_x if isinstance(predict_min_x, float) else self.x.min()
        self.num_predictions = num_predictions if isinstance(num_predictions, int) else len(self.x)
        self.num_predictions = 2 if self.num_predictions < 2 else self.num_predictions

        self.equation = self._set_fit_equation(type=fit_type)

        self.fit_results = {}  # Stores fit results

    def run(self):
        self.fit_results = self._fit(df=self.df.copy())

    def get_results(self):
        return self.fit_results

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
        else:
            equation = self._fit_quadratic
        return equation

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
        c = fit_params_opt[2]

        # Calc r2
        kwargs = dict(x=x, a=a, b=b, c=c)
        fit_r2 = \
            1.0 - (sum((y - self.equation(**kwargs)) ** 2) / ((len_y - 1.0) * np.var(y, ddof=1)))

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
