"""
SET TO: CONDITIONAL VALUE REPLACEMENT
======================================

Set data to missing, specific values, or thresholds based on conditions.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from pandas import Series

from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.console import info, detail
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

    Example:
        See `examples/pkgs/preprocessing/corrections/correction_set_exact_values_to_missing.py`
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

    info(f"Correction: set exact values to missing")
    info(f"  Variable: {series.name}")
    info(f"  Number of records set to missing: {n_vals}")

    if verbose > 0:
        detail(f"  Locations of records set to missing: {locs}")

    # Plot
    if showplot:
        quickplot([series, series_corr], subplots=True, showplot=showplot,
                  title=f"Set exact values in {series.name} to missing values")

    return series_corr


def setto_value(series: Series, dates: list, value: float = 0, verbose: int = 0):
    """
    Set time range(s) to specific value

    Args:
        series: time series
        dates: list, can be given as a mix of strings and lists that
            contain the date(times) of records that should be removed
            Example
            *dates=['2022-06-30 23:58:30', ['2022-06-05 00:00:30', '2022-06-07 14:30:00']]*
            will select the record for '2022-06-30 23:58:30' and all records between
            '2022-06-05 00:00:30' (inclusive) and '2022-06-07 14:30:00' (inclusive).
            * This also works when providing only the date, e.g.
            dates=['2006-05-01', '2006-07-18'] will select all data points between
            2006-05-01 and 2006-07-18.
        value: the value filled in for all records that fall within the specified time range(s)
        verbose: more text output to console if verbose > 0

    Returns:
        *series* with records in time range *dates* set to *value*

    Example:
        See `examples/pkgs/preprocessing/corrections/correction_setto_value.py`
    """
    series_corr = series.copy()
    for date in dates:
        if isinstance(date, str):
            # Neat solution: even though here only data for a single datetime
            # is removed, the >= and <= comparators are used to avoid an error
            # in case the datetime is not found in the flag.index
            date = (series_corr.index >= date) & (series_corr.index <= date)
            series_corr.loc[date] = value
        elif isinstance(date, list):
            _dates = (series_corr.index >= date[0]) & (series_corr.index <= date[1])
            series_corr.loc[_dates] = value
    if verbose > 0:
        info(f"Records in time range {dates} were set to value {value}.")
    return series_corr


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

    Example:
        See `examples/pkgs/preprocessing/corrections/correction_setto_threshold.py`
    """
    outname = series.name
    series.name = "input_data"

    # Create empty flag
    flag = pd.Series(index=series.index, data=np.nan)

    # Detect values over threshold
    if type == 'max':
        over_thres_ix = series > threshold
        range_ok_ix = series <= threshold
    elif type == 'min':
        over_thres_ix = series < threshold
        range_ok_ix = series >= threshold
    else:
        raise ValueError(f"type must be 'min' or 'max', got {type!r}.")

    flag.loc[over_thres_ix] = 1
    flag.loc[range_ok_ix] = 0

    info(f"QA/QC set to threshold value")
    info(f"  Variable: {series.name}")
    if type == 'max':
        info(f"  Accepted: {range_ok_ix.sum()} values below max threshold of {threshold}")
        info(f"  Corrected: {over_thres_ix.sum()} values above max threshold of {threshold} were set to {threshold}")
    if type == 'min':
        info(f"  Accepted: {range_ok_ix.sum()} values above min threshold of {threshold}")
        info(f"  Corrected: {over_thres_ix.sum()} values below min threshold of {threshold} were set to {threshold}")

    corrected_ix = flag == 1
    series_corr = series.copy()
    series_corr.loc[corrected_ix] = threshold
    series_corr = series_corr.rename(outname)

    # Plot
    if showplot:
        quickplot([series, series_corr], subplots=True, showplot=showplot,
                  title=f"Set {series.name} to {type} threshold {threshold}")

    return series_corr
