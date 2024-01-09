from functools import wraps

import pandas as pd


def repeater(cls):
    """Repeater function for outlier detection.

    Runs an outlier detection class multiple times and stores
    results from each iteration in a dataframe.

    """

    # @wraps creates yet another wrapper around a decorated function that restores its type as a function
    # while preserving the docstring.
    # https://stackoverflow.com/questions/72492374/how-to-make-python-help-function-work-well-with-decorators
    @wraps(cls)
    def wrapper(*args, **kwargs) -> pd.DataFrame:

        iterr = 0
        results_df = pd.DataFrame()
        n_outliers = 9999

        while n_outliers > 0:
            iterr += 1
            results = cls(*args, **kwargs)  # Init outlier class
            results.calc()  # Calculate flag
            results_df = _add_iterdata(df=results_df, results=results, iterr=iterr)
            n_outliers = results.flag.sum()  # Count how many times flag = 1 (flagged outlier)

            # If repeated until no outliers left, the series for next iteration
            # has to be the filtered series from current iteration.
            if kwargs['repeat']:
                kwargs['series'] = results.filteredseries
            else:
                break

        return results_df

    return wrapper


def _add_iterdata(df: pd.DataFrame, results, iterr) -> pd.DataFrame:
    """Add filtered series and flag for current iteration."""
    df[results.filteredseries.name] = results.filteredseries
    flagname = results.flag.name
    addflagname = flagname.replace('_TEST', f'_ITER{iterr}_TEST')
    df[addflagname] = results.flag
    return df
