import numpy as np
import pandas as pd
from pandas import Series

from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def setto_threshold(series: Series,
                    threshold: float,
                    type: str,
                    showplot: bool = False) -> Series:
    """
    Set values above or below a threshold value to threshold value

    Args:
        series: Data for variable that is corrected
        threshold: Threshold value
        type: `min` sets series values below *threshold* to *threshold*,
            'max' sets series values above *threshold* to *threshold*
        showplot: Show plot

    Returns:
        Corrected series
    """
    outname = series.name
    series.name = "input_data"

    # Create empty flag
    flag = pd.Series(index=series.index, data=np.nan)

    # Detect values over threshold
    over_thres_ix = range_ok_ix = None
    if type == 'max':
        over_thres_ix = series > threshold
        range_ok_ix = series <= threshold
    if type == 'min':
        over_thres_ix = series < threshold
        range_ok_ix = series >= threshold

    flag.loc[over_thres_ix] = 1
    flag.loc[range_ok_ix] = 0

    print(f"QA/QC set to threshold value")
    print(f"    Variable: {series.name}")
    if type == 'max':
        print(f"    Accepted → {range_ok_ix.sum()} values below max threshold of {threshold}")
        print(
            f"    Corrected → {over_thres_ix.sum()} values above max threshold of {threshold} were set to {threshold}")
    if type == 'min':
        print(f"    Accepted → {range_ok_ix.sum()} values above min threshold of {threshold}")
        print(
            f"    Corrected → {over_thres_ix.sum()} values below min threshold of {threshold} were set to {threshold}")

    corrected_ix = flag == 1
    series_corr = series.copy()
    series_corr.loc[corrected_ix] = threshold
    series_corr.rename(outname, inplace=True)

    # Plot
    if showplot:
        quickplot([series, series_corr], subplots=True, showplot=showplot,
                  title=f"Set {series.name} to {type} threshold {threshold}")

    return series_corr
