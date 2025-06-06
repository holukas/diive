import numpy as np
import pandas as pd
from pandas import Series

from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def set_exact_values_to_missing(series: Series,
                                values: list,
                                showplot: bool = False,
                                verbose: int = 0) -> Series:
    """
    Set records matching *values* to missing values.

    Args:
        series: Data for variable that is corrected
        values: List of floats that will be set to missing values
        showplot: Show plot
        verbose: Verbosity level

    Returns:
        Corrected series
    """
    outname = series.name
    series.name = "input_data"

    # Create empty flag
    flag = pd.Series(index=series.index, data=np.nan)

    # Create flag that indicates where records match one of the values
    for val in values:
        locs = series == val
        flag[locs] = 1

    # Set flag to zero for all other records
    flag = flag.fillna(0)

    # Apply flag: set records to missing
    setto_missing_ix = flag == 1
    series_corr = series.copy()
    series_corr.loc[setto_missing_ix] = np.nan
    series_corr = series_corr.rename(outname)

    # Indexes where records were set to missing
    locs = flag[flag == 1].index.tolist()

    # Number of values set to missing
    n_vals = int(flag.sum())

    print(f"Correction: set exact values to missing")
    print(f"    Variable: {series.name}")
    print(f"    Number of records set to missing: {n_vals}")

    if verbose > 0:
        print(f"    Locations of records set to missing: {locs}")

    # Plot
    if showplot:
        quickplot([series, series_corr], subplots=True, showplot=showplot,
                  title=f"Set exact values in {series.name} to missing values")

    return series_corr


def _example():
    series = [1, 2, 0, 4, 5, 6, 7, 0, 9, 10]
    series = pd.Series(series)
    series.name = "testdata"
    series_corr = set_exact_values_to_missing(series=series, values=[0, 1, 10], showplot=True)
    print(series_corr)


if __name__ == '__main__':
    _example()
