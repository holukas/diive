"""
DATA FUNCTIONS: FITS
====================

This module is part of the diive library:
https://github.com/holukas/diive

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
        df.groupby('group', observed=False).agg(
            {'mean', 'median', 'max', 'min', 'count', 'std', q25, q75})

    # print(numvals_in_group)

    # Bins info
    grouped_df['BIN_START'] = bins[0:-1]

    return grouped_df


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


def fit_to_bins_polyreg(df, x_col, y_col, degree, bin_col=None):
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

# if __name__ == '__main__':
#     from diive.configs.exampledata import load_exampledata_parquet
#
#     df_orig = load_exampledata_parquet()
#     df_orig = df_orig.loc[df_orig.index.year == 2021].copy()
#
#     # Variables
#     vpd_col = 'VPD_f'
#     nee_col = 'NEE_CUT_REF_f'
#     xcol = vpd_col
#     ycol = nee_col
#
#     # Select daytime data between May and September 1997-2021
#     maysep_dt_df = df_orig.loc[(df_orig.index.month >= 6) & (df_orig.index.month <= 6)].copy()
#     # maysep_dt_df = maysep_dt_df.loc[maysep_dt_df['PotRad_CUT_REF'] > 20]
#
#     # Convert units
#     maysep_dt_df[vpd_col] = maysep_dt_df[vpd_col].multiply(0.1)  # hPa --> kPa
#     maysep_dt_df[nee_col] = maysep_dt_df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
#     x_units = "kPa"
#     y_units = "gCO_{2}\ m^{-2}\ 30min^{-1}"
#     xlabel = f"Half-hourly VPD ({x_units})"
#     ylabel = f"{ycol} (${y_units}$)"
#
#     bf = BinFitterCP(
#         df=maysep_dt_df,
#         # n_bootstraps=2,
#         xcol=xcol,
#         ycol=ycol,
#         predict_max_x=maysep_dt_df[xcol].max(),
#         predict_min_x=maysep_dt_df[xcol].min(),
#         n_predictions=1000,
#         n_bins_x=20,
#         bins_y_agg='mean',
#         fit_type='quadratic_offset'
#     )
#     bf.run()
#     fit_results = bf.fit_results
#     # bf.showplot_binfitter(highlight_year=None, xlabel=xlabel, ylabel=ylabel)
#
#     from diive.core.plotting.fitplot import fitplot
#
#     # Fitplot
#     fig = plt.figure(facecolor='white', figsize=(9, 9), dpi=100)
#     gs = gridspec.GridSpec(1, 1)  # rows, cols
#     # gs.update(wspace=0, hspace=0, left=.2, right=.8, top=.8, bottom=.2)
#     ax = fig.add_subplot(gs[0, 0])
#     line_xy_gpp, line_fit_gpp, line_fit_ci_gpp, line_fit_pb_gpp, line_highlight = \
#         fitplot(
#             ax=ax,
#             label='year',
#             flux_bts_results=fit_results,
#             # flux_bts_results=bts_fit_results[0],
#             alpha=1,
#             edgecolor='r',
#             color='r',
#             color_fitline='r',
#             show_prediction_interval=False,
#             size_scatter=90,
#             fit_type='quadratic_offset',
#             highlight_year=None
#         )
#
#     fig.tight_layout()
#     fig.show()
