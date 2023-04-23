from statsmodels.formula import api as smf


# import statsmodels.formula.api as smf

def linear(df):
    """
    Perform simple linear regression

    df contains only two columns, with the variable y that is predicted in
    column 0, and the independent variable X1 in column 1.

    Parameters
    ----------
    df: DataFrame
        Contains only 2 columns, used for y and X1.

    Returns
    -------
    k: float
        Slope of regression equation.

    d: float
        Intercept of regression equation.

    fitted_values: Series
        Predicted values, contains pandas.Index.

    rsquared: float
        Rsquared for regression equation.

    rsquared_adj: float
        Adjusted rsquared for regression equation.

    """
    # Data
    df_no_nan = df.copy()
    df_no_nan.dropna(inplace=True)
    df_no_nan.columns = ['y', 'X1']

    # Formula, regression equation
    formula_y = df_no_nan.columns[0]  # To be predicted
    formula_X1 = df_no_nan.columns[1]  # Independent var 1, predictor
    formula_string = '{} ~ {}'.format(formula_y, formula_X1)

    model = smf.ols(formula_string, data=df_no_nan)  # -1 would remove the intercept
    results = model.fit()
    # print(results.summary())

    # Coefficients
    d = results.params['Intercept']
    k = results.params['X1']
    fitted_values = results.fittedvalues
    rsquared = results.rsquared
    rsquared_adj = results.rsquared_adj

    return k, d, fitted_values, rsquared, rsquared_adj, results
