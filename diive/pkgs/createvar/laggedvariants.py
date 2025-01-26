from pandas import DataFrame


def lagged_variants(df: DataFrame,
                    lag: list[int, int],
                    stepsize: int = 1,
                    exclude_cols: list = None,
                    verbose: int = 0) -> DataFrame:
    """Create lagged variants of variables

    Shifts all records in *df* by *lag* records and stores he lagged variants
    as new columns.

    For example, lagging the two variables `TA` and `SW_IN` with the settings *lag=[-2, 1]*
    and *stepsize=1* creates the new variables `TA-2` (two records before `TA`),
    `TA-1` (one record before `TA`), `TA+1` (one record after `TA`), `SW_IN-2`, `SW_IN-1`,
    and `SW_IN+1`. Note that the minus sign means "before", the plus sign means "after".

    Can be used to investigate correlations between a scalar and the preceding or subsequent
    records of another scalar.

    Note:
        The timestamp index of `df` must be regular and complete, i.e., all timestamps of the
        respective time resolution must be present. Otherwise shifting variables by x records
        might lead to undesirable results.

    Args:
        df: dataframe that contains variables that will be lagged
        lag: list of integers given as number or records, defining the range of generated lag times
            For example lag=[-3, 2] and stepsize=1 will generated lagged
            variants -3, -2, -1, +1 and +2
        stepsize: stepsize between the different lagged variants given as number of records
            For example lag=[-8, 4] and stepsize=2 will generated lagged
            variants -8, -6, -4, -2 and +2
        exclude_cols: list of column names, these variables will not be lagged
        verbose: if *True*, print more output to console

    Returns:
        input dataframe with added lagged variants
    """

    if len(df.columns) == 1:
        if df.columns[0] in exclude_cols:
            raise Exception(f"(!) No lagged variants can be created "
                            f"because there is only one single column in the dataframe "
                            f"({df.columns[0]}) and the same column is also defined in "
                            f"the exclude list (exclude_cols={exclude_cols}). "
                            f"This means there are no data left to lag.")
        return df

    if not isinstance(lag, list):
        raise Exception(f"(!) Error in lag={lag}: No lagged variables can be created "
                        f"because lag is not given as a list, e.g. lag=[-10, -1]. "
                        f"(it was given as lag={lag})")

    if len(lag) != 2:
        raise Exception(f"(!) Error in lag={lag}: No lagged variables can be created "
                        f"because lag must be given as a list with two elements, "
                        f"e.g. lag=[-10, -1]. (it was given as lag={lag})")

    for _lag in lag:
        if not isinstance(_lag, int):
            raise TypeError(f"(!) Error in lag={lag}: No lagged variables can be created "
                            f"because {_lag} is not an integer.")

    _included = []
    _excluded = []

    lagsteps = range(lag[0], lag[1] + 1, stepsize)

    for col in df.columns:
        if exclude_cols:

            if col in exclude_cols:
                _excluded.append(col)
                continue

            for lagstep in lagsteps:
                if lagstep < 0:
                    stepname = f".{col}{lagstep}"
                    _shift = abs(lagstep)
                elif lagstep > 0:
                    stepname = f".{col}+{lagstep}"
                    _shift = -lagstep
                else:
                    # skip if lagstep = 0
                    continue

                # Shifting data creates NaNs in time series, which can cause issues
                # with some machine learning algorithms
                # To handle this, the new gap(s) is/are filled with the nearest value
                n_missing_vals_before = int(df[col].isnull().sum())
                df[stepname] = df[col].shift(_shift)
                n_missing_vals_after = int(df[stepname].isnull().sum())
                if n_missing_vals_before == 0 and n_missing_vals_after > 0:
                    df[stepname] = df[stepname].bfill(limit=n_missing_vals_after)
                    df[stepname] = df[stepname].ffill(limit=n_missing_vals_after)
            _included.append(col)

    if verbose:
        print(f"++ Added new columns with lagged variants for: {_included} (lags between {lag[0]} and {lag[1]} "
              f"with stepsize {stepsize}), no lagged variants for: {_excluded}. "
              f"Shifting the time series created gaps which were then filled with the nearest value.")
    return df
