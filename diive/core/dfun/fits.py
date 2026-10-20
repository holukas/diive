"""
DATA FUNCTIONS: FITS
====================

Binning helpers for fits with confidence interval (CI) and prediction interval (PI).

Part of the diive library: https://github.com/holukas/diive
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

import pandas as pd

# from gui import plotfuncs
from diive.core.dfun.stats import q25, q75


def groupagg(df, num_bins, bin_col) -> pd.DataFrame:
    """Bin *bin_col* into *num_bins* quantile groups and return per-bin aggregates.

    Adds the bin labels to *df* as column ``group`` (in place).
    """
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
        df.groupby('group', observed=False).agg(
            {'mean', 'median', 'max', 'min', 'count', 'std', q25, q75})

    # print(numvals_in_group)

    # Bins info
    grouped_df['BIN_START'] = bins[0:-1]

    return grouped_df


def fit_to_bins_linreg(df, x_col, y_col, bin_col):
    """Fit a linear regression of *y_col* on *x_col* within each bin of *bin_col*."""
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


def fit_to_bins_polyreg(df, x_col, y_col, degree, bin_col=None):
    """Fit a polynomial regression of degree *degree* of *y_col* on *x_col*, optionally per bin."""
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

    X = _df[x_col].values.reshape(-1, 1)  # Attributes (independent)
    y = _df[y_col].values.reshape(-1, 1)  # Labels (dependent, predicted)
    # X = _df[x_col][bin_col].values.reshape(-1, 1)  # Attributes (independent)
    # y = _df[y_col][bin_col].values.reshape(-1, 1)  # Labels (dependent, predicted)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin = LinearRegression()
    lin.fit(X_poly, y)

    y_pred = lin.predict(poly.fit_transform(X))

    predicted_col = 'predicted'
    # predicted_col = (y_col[0], y_col[1], 'predicted')
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
