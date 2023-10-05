from pandas import Series


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
        print(f"Records in time range {dates} were set to value {value}.")
    return series_corr
