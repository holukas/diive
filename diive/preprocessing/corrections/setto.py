"""
SET TO: CONDITIONAL VALUE REPLACEMENT
======================================

Set data to missing, specific values, or thresholds based on conditions.

Part of the diive library: https://github.com/holukas/diive
"""

import numbers

import numpy as np
import pandas as pd
from pandas import Series

from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.console import info, detail


def _fits_value(series: Series, value) -> Series:
    """Return *series* as float64 if its integer dtype cannot hold *value*.

    pandas 3 raises when a non-integer value (e.g. 2.5) is assigned into an
    integer series, e.g. an InfluxDB integer field. Integer values keep the
    integer dtype, and float series are returned unchanged.
    """
    if pd.api.types.is_integer_dtype(series.dtype) and not float(value).is_integer():
        return series.astype(float)
    return series


def set_exact_values_to_missing(series: Series,
                                values: list,
                                showplot: bool = False,
                                verbose: int | bool | None = None) -> Series:
    """
    Set records matching *values* to missing values.

    A record matches when it equals the value within a relative tolerance of
    1e-6 (``np.isclose`` with ``rtol=1e-6, atol=0``), so data stored as float32
    still match: float32 0.1 is 0.10000000149 as float64 and does not equal
    0.1. The tolerance is below the resolution of measurements (-9999 matches
    values within 0.01 of it). 0 matches only 0.

    Args:
        series: Data for variable that is corrected
        values: List of numbers that will be set to missing values. Any other
            entry (None, pd.NA, a string such as '-9999', a bool) raises a
            ValueError before anything is changed, because a value that is
            silently skipped would leave e.g. -9999 in the data. NaN is a
            number and matches nothing.
        showplot: Show plot
        verbose: Verbosity for the console helpers. The report appears at
            ``VERBOSE_PROGRESS`` (2) and above, the locations of the records
            at ``VERBOSE_DEBUG`` (3). 0 prints nothing. None (default) uses the
            level set with ``set_verbosity()``.

    Returns:
        Corrected series

    Example:
        See `examples/preprocessing/corrections/correction_set_exact_values_to_missing.py`
    """
    if isinstance(values, (str, bytes)) or not np.iterable(values):
        raise ValueError(f"values must be a list of numbers, got {values!r}.")
    invalid = [v for v in values if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)]
    if invalid:
        raise ValueError(f"values must be numbers (e.g. -9999, not '-9999'), got {invalid!r}.")

    outname = series.name
    # Renamed copy, not an in-place rename: the label is only needed so the
    # quickplot panels distinguish input from output, and renaming the parameter
    # would leave the caller's own series called "input_data".
    work = series.rename("input_data")

    # Create empty flag
    flag = pd.Series(index=work.index, data=np.nan)

    # Create flag that indicates where records match one of the values. Relative
    # tolerance so float32 storage matches, atol=0 so 0 matches only 0.
    values_float = work.to_numpy(dtype=float, na_value=np.nan)
    for val in values:
        flag[np.isclose(values_float, val, rtol=1e-6, atol=0)] = 1

    # Set flag to zero for all other records
    flag = flag.fillna(0)

    # Apply flag: set records to missing
    setto_missing_ix = flag == 1
    series_corr = work.copy()
    series_corr.loc[setto_missing_ix] = np.nan
    series_corr = series_corr.rename(outname)

    # Indexes where records were set to missing
    locs = flag[flag == 1].index.tolist()

    # Number of values set to missing
    n_vals = int(flag.sum())

    info(f"Correction: set exact values to missing", verbose=verbose)
    info(f"  Variable: {outname}", verbose=verbose)
    info(f"  Number of records set to missing: {n_vals}", verbose=verbose)
    detail(f"  Locations of records set to missing: {locs}", verbose=verbose)

    # Plot
    if showplot:
        quickplot([work, series_corr], subplots=True, showplot=showplot,
                  title=f"Set exact values in {outname} to missing values")

    return series_corr


def setto_value(series: Series, dates: list, value: float = 0, verbose: int = 0):
    """
    Set time range(s) to specific value

    Args:
        series: time series
        dates: list of the date(times) of records that are set to *value*. A
            string is a single date(time); a range is a nested ``[start, end]``
            list, both ends inclusive. A bare date covers the whole day.
            Example
            *dates=['2022-06-30 23:58:30', ['2022-06-05 00:00:30', '2022-06-07 14:30:00']]*
            will select the record for '2022-06-30 23:58:30' and all records between
            '2022-06-05 00:00:30' (inclusive) and '2022-06-07 14:30:00' (inclusive).
            A flat list such as *dates=['2006-05-01', '2006-07-18']* selects the
            two days 2006-05-01 and 2006-07-18; *dates=[['2006-05-01', '2006-07-18']]*
            selects all records from 2006-05-01 to the end of 2006-07-18.
        value: the value filled in for all records that fall within the specified time range(s)
        verbose: more text output to console if verbose > 0

    Returns:
        *series* with records in time range *dates* set to *value*
        (float64 if *series* is integer and *value* is not a whole number)

    Example:
        See `examples/preprocessing/corrections/correction_setto_value.py`
    """
    series_corr = _fits_value(series, value).copy()
    for date in dates:
        # Label slicing matches partial strings, so a bare date covers the whole
        # day (a >= / <= comparison would stop at midnight), as in ManualRemoval.
        # A timestamp that is not in the index selects nothing instead of raising.
        if isinstance(date, str):
            series_corr.loc[date:date] = value
        elif isinstance(date, list):
            series_corr.loc[date[0]:date[1]] = value
    if verbose > 0:
        info(f"Records in time range {dates} were set to value {value}.")
    return series_corr


def setto_threshold(series: Series,
                    threshold: float,
                    type: str,
                    showplot: bool = False,
                    verbose: int | bool | None = None) -> Series:
    """
    Set values above or below a threshold value to threshold value

    Args:
        series: Data for variable that is corrected
        threshold: Threshold value
        type: `min` sets series values below *threshold* to *threshold*,
            'max' sets series values above *threshold* to *threshold*
        showplot: Show plot
        verbose: Verbosity for the console helpers. The report appears at
            ``VERBOSE_PROGRESS`` (2) and above, 0 prints nothing. None (default)
            uses the level set with ``set_verbosity()``.

    Returns:
        Corrected series (float64 if *series* is integer and *threshold* is
        not a whole number)

    Example:
        See `examples/preprocessing/corrections/correction_setto_threshold.py`
    """
    outname = series.name
    # Renamed copy, not an in-place rename: the label is only needed so the
    # quickplot panels distinguish input from output, and renaming the parameter
    # would leave the caller's own series called "input_data".
    work = series.rename("input_data")

    # Create empty flag
    flag = pd.Series(index=work.index, data=np.nan)

    # Detect values over threshold
    if type == 'max':
        over_thres_ix = work > threshold
        range_ok_ix = work <= threshold
    elif type == 'min':
        over_thres_ix = work < threshold
        range_ok_ix = work >= threshold
    else:
        raise ValueError(f"type must be 'min' or 'max', got {type!r}.")

    flag.loc[over_thres_ix] = 1
    flag.loc[range_ok_ix] = 0

    info(f"QA/QC set to threshold value", verbose=verbose)
    info(f"  Variable: {outname}", verbose=verbose)
    if type == 'max':
        info(f"  Accepted: {range_ok_ix.sum()} values below max threshold of {threshold}", verbose=verbose)
        info(f"  Corrected: {over_thres_ix.sum()} values above max threshold of {threshold} were set to {threshold}",
             verbose=verbose)
    if type == 'min':
        info(f"  Accepted: {range_ok_ix.sum()} values above min threshold of {threshold}", verbose=verbose)
        info(f"  Corrected: {over_thres_ix.sum()} values below min threshold of {threshold} were set to {threshold}",
             verbose=verbose)

    corrected_ix = flag == 1
    series_corr = _fits_value(work, threshold).copy()
    series_corr.loc[corrected_ix] = threshold
    series_corr = series_corr.rename(outname)

    # Plot
    if showplot:
        quickplot([work, series_corr], subplots=True, showplot=showplot,
                  title=f"Set {outname} to {type} threshold {threshold}")

    return series_corr
