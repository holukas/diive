"""
OUTLIER DETECTION: repeater wrapper
===================================

This module is part of the diive library:
https://github.com/holukas/diive

"""
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
            results._calc()  # Calculate flag
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


def _add_iterdata(df: pd.DataFrame, results, iterr: int) -> pd.DataFrame:
    """"Add flag and filtered series for current iteration.

    Args:
        df: dataframe that collects the output (flags and filtered series) from
            all iterations and all outlier methods that were selected. Note that
            e.g. the class `StepwiseOutlierDetection` applies multiple outlier
            detection methods in sequence, and each method is run multiple times
            via the  @repeater wrapper.
        results: class property of the respective outlier method that stores the
            outlier flag in results.flag and the correspondingly filtered time series
             in results.filteredseries. The flag stores accepted (ok) values with flag=0,
             rejected values are indicated with flag=2. The filtered series contains
             data of the original (unfiltered) time series with rejected values set to missing.
        iterr: current iteration

    Returns:
        dataframe with new results from this iteration added

    """
    df[results.filteredseries.name] = results.filteredseries
    flagname = results.flag.name
    addflagname = flagname.replace('_TEST', f'_ITER{iterr}_TEST')
    df[addflagname] = results.flag
    return df
