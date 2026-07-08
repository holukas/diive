"""Resample time series to coarser aggregates."""
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series
from pandas.tseries.frequencies import to_offset

from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import convert_series_timestamp_to_middle
from diive.core.utils.console import info
from diive.core.utils.prints import ConsoleOutputDecorator


def resample_to_monthly_agg_matrix(series: pd.Series, agg: str, ranks: bool = False) -> pd.DataFrame:
    """Resample to monthly aggregation across years, store results in a matrix.

    Args:
        series: Time series with timestamp.
        agg: Method for monthly aggregation, e.g. *agg='mean'*.
            Allowed methods: 'mean', 'median', 'sum', 'max', 'min', 'std', 'skew'.
        ranks: Returns ranks of aggregation values in the matrix.
            For example, if *agg='mean'* and *ranks=True*, then the highest value
            in each month across years is attributed rank 1, the second highest rank 2, etc.
            For measurements such as air temperature, rank 1 means that the respective month
            was the warmest across all years.

    Returns:
        Matrix with resampled data. The resulting dataframe has YEAR as index and MONTH as columns.
    """
    series.name = 'data' if not series.name else series.name
    df = pd.DataFrame(series)
    df['MONTH'] = df.index.month
    df['YEAR'] = df.index.year
    monthly_agg = df.groupby(['YEAR', 'MONTH'])[series.name].agg(agg).unstack()

    # Calculate ranks per month
    def rank_monthly_aggs(col):
        return col.rank(method='dense', ascending=False)

    if ranks:
        monthly_agg = monthly_agg.apply(rank_monthly_aggs, axis=0)

    return monthly_agg


def resample_to_daily_agg(series: Series,
                          agg: str = 'mean',
                          mincounts_perc: float = 0.0) -> Series:
    """Resample a time series to one aggregated value per calendar day.

    Aggregates a (typically sub-daily) time series down to daily resolution,
    e.g. a daily-mean time series over the record. Days that hold too few
    records can be dropped via ``mincounts_perc``.

    Args:
        series: Time series with a ``DatetimeIndex``.
        agg: Aggregation method, anything pandas' ``Series.agg`` accepts
            (e.g. ``'mean'``, ``'sum'``, ``'median'``, ``'min'``, ``'max'``,
            ``'std'``). Defaults to ``'mean'``.
        mincounts_perc: Minimum fraction (0-1) of the fullest day's record count
            that a day must reach to be kept. ``0`` (default) keeps every day
            that has at least one value; ``0.9`` keeps only days at least 90 %
            complete relative to the most-populated day.

    Returns:
        Daily-aggregated Series, indexed by day (timestamp at day start), with
        the input series' name preserved.

    Raises:
        TypeError: If ``series`` does not have a ``DatetimeIndex``.

    See Also:
        examples/times/times_resample_daily.py — daily aggregation with several methods
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("resample_to_daily_agg requires a series with a DatetimeIndex.")

    daily = series.resample('D')
    counts = daily.count()
    agg_ser = daily.agg(agg)

    # Keep days with enough records: relative to the fullest day, but always at
    # least one value (a day of all-NaN aggregates to NaN and is dropped anyway).
    maxcounts = counts.max()
    mincounts = max(1, int(maxcounts * mincounts_perc)) if maxcounts else 1
    agg_ser = agg_ser[counts >= mincounts]

    agg_ser.name = series.name
    return agg_ser


@ConsoleOutputDecorator()
def resample_series_to_freq(series: Series,
                            to_freqstr: str = '30min',
                            agg: Literal['mean', 'sum'] = 'mean',
                            mincounts_perc: float = .9,
                            output_timestamp_shows: Literal['middle', 'end'] = 'end') -> Series:
    """Downsample data to an arbitrary (lower) time resolution.

    Input data must have timestamp showing the END of the time period.
    Before resampling, the timestamp is converted to show the MIDDLE
    of the time period. After resampling, the timestamp shows again the
    END of the time period.

    Using the selected aggregation method and while also considering the
    minimum required values in the aggregation time window.

    Args:
        series: regular-frequency time series with a TIMESTAMP_END index.
        to_freqstr: target pandas offset alias, e.g. '30min', '10min', '1h'.
            Must be a *lower* time resolution than the source (downsampling only).
        agg: aggregation method ('mean' or 'sum').
        mincounts_perc: minimum fraction (0-1) of the maximum possible records
            an interval must have for its aggregate to be kept (else NaN).
        output_timestamp_shows: 'end' (default) or 'middle' of the period.

    Note regarding .resample:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        closed : {‘right’, ‘left’}, default None
            Which side of bin interval is closed. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

        label : {‘right’, ‘left’}, default None
            Which bin edge label to label bucket with. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

    See here for a table of updated date offsets in pandas 2.2+:
    https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects

    https://stackoverflow.com/questions/48340463/how-to-understand-closed-and-label-arguments-in-pandas-resample-method
        closed='right' =>  ( 3:00, 6:00 ]  or  3:00 <  x <= 6:00
        closed='left'  =>  [ 3:00, 6:00 )  or  3:00 <= x <  6:00

    """

    # 'T' is the deprecated pandas minute alias (removed in pandas 4); normalize
    # e.g. '30T' -> '30min', '10T' -> '10min' so the old forms keep working.
    if to_freqstr.endswith('T'):
        to_freqstr = f"{to_freqstr[:-1]}min"

    current_freq = to_offset(series.index.freqstr)
    requested_freq = to_offset(to_freqstr)

    # Timestamp must be regular
    if not series.index.freq:
        raise NotImplementedError("Error during resampling: Irregular timestamps are not supported.")

    # Requested frequency must be larger than data freq
    if current_freq > requested_freq:
        raise NotImplementedError(
            f"Error during resampling: "
            f"Upsampling not allowed. "
            f"Target frequency {to_freqstr} must be lower time resolution than "
            f"source frequency {series.index.freqstr}.")

    # Already at the target resolution (e.g. processed 30MIN data resampled to
    # 30min): aggregating would be a no-op, so just return the series in the
    # requested timestamp convention.
    if current_freq == requested_freq:
        info(f"Data already at {to_freqstr} resolution; no resampling needed.")
        out = series.copy()
        if output_timestamp_shows == 'middle':
            out = convert_series_timestamp_to_middle(data=out)
        return out

    _series = series.copy()

    # Make middle timestamp, for correct resampling
    _series = convert_series_timestamp_to_middle(data=_series)

    info(f"Resampling data from {current_freq.freqstr} to {requested_freq.freqstr} frequency ...")

    # Check maximum number of counts per aggregation interval
    # Needed to calculate the required minimum number of counts from
    # 'mincounts_perc', which is a relative threshold.
    maxcounts = pd.Series(index=_series.index, data=1)  # Dummy series of 1s
    maxcounts = maxcounts.resample(to_freqstr, label='right').count().max()
    # maxcounts = maxcounts.resample(to_freqstr, label=label, closed=closed).count().max()
    mincounts = int(maxcounts * mincounts_perc)

    # Require minimum 1 value
    # Relevant e.g. for 10MIN data (3 values per half-hour)
    mincounts = 1 if mincounts < 3 else mincounts

    # Aggregation
    resampled_ser = _series.resample(to_freqstr, label='right')  # default: closed='left'
    # resampled_df = _series.resample(to_freqstr, label=label, closed=closed)
    agg_counts_ser = resampled_ser.count()  # Count aggregated values, always needed
    agg_ser = resampled_ser.agg(agg)

    # Timestamp index shows end after resampling b/c label='right'
    agg_counts_ser.index.name = 'TIMESTAMP_END'
    agg_ser.index.name = 'TIMESTAMP_END'

    # Keep aggregates with enough values
    filter_min = agg_counts_ser >= mincounts
    agg_ser = agg_ser[filter_min].copy()

    # Re-assign 30MIN resolution as freq
    agg_ser = agg_ser.asfreq(to_freqstr)

    # Sanitize resampled timestamp index
    if not agg_ser.empty:
        if output_timestamp_shows == 'middle':
            agg_ser = TimestampSanitizer(data=agg_ser).get()
        elif output_timestamp_shows == 'end':
            agg_ser = TimestampSanitizer(data=agg_ser, output_middle_timestamp=False).get()

    return agg_ser


def resample_series_to_30MIN(series: Series,
                             to_freqstr: Literal['30T', '30min'] = '30min',
                             agg: Literal['mean', 'sum'] = 'mean',
                             mincounts_perc: float = .9,
                             output_timestamp_shows: Literal['middle', 'end'] = 'end') -> Series:
    """Downsample data to 30-minute time resolution.

    Thin wrapper around :func:`resample_series_to_freq` kept for back-compat;
    use that function directly for other resolutions. See it for full details.
    """
    if to_freqstr == '30T':
        to_freqstr = '30min'
    if to_freqstr != '30min':
        raise NotImplementedError(
            "resample_series_to_30MIN only resamples to 30min; "
            "use resample_series_to_freq for other resolutions.")
    return resample_series_to_freq(
        series=series, to_freqstr='30min', agg=agg,
        mincounts_perc=mincounts_perc, output_timestamp_shows=output_timestamp_shows)


def diel_cycle(series: Series,
               mincounts: int = 1,
               mean: bool = True,
               std: bool = True,
               median: bool = False,
               quantiles: bool = False,
               minmax: bool = False,
               each_month: bool = False,
               ) -> pd.DataFrame:
    """Calculate diel cycles grouped by time

    Args:
        quantiles: Also compute the 25th/75th percentiles (columns ``q25``/``q75``).
        minmax: Also compute the per-time-of-day minimum/maximum (``min``/``max``).

    See Also:
        examples/visualization/dielcycle.py — Diurnal cycle visualization with DielCycle class
    """
    from diive.core.dfun.stats import q25, q75

    # Build list with agg strings
    aggstr = ['count']  # Available values always counted
    aggstr.append('mean') if mean else aggstr
    aggstr.append('std') if std else aggstr
    aggstr.append('median') if median else aggstr
    if quantiles:
        aggstr.extend([q25, q75])  # -> columns 'q25'/'q75' (function names)
    if minmax:
        aggstr.extend(['min', 'max'])

    if each_month:
        aggs = series.groupby([series.index.month, series.index.time]).agg(aggstr)
        # aggs = aggs.unstack()
        # aggs = aggs.transpose()
    else:
        aggs = series.groupby([series.index.time]).agg(aggstr)

    if mean and std:
        aggs['mean+sd'] = aggs['mean'] + aggs['std']
        aggs['mean-sd'] = aggs['mean'] - aggs['std']

    if median and std:
        aggs['median+sd'] = aggs['median'] + aggs['std']
        aggs['median-sd'] = aggs['median'] - aggs['std']

    remove = aggs['count'] < mincounts
    aggs[remove] = np.nan

    # df = pd.DataFrame(series)
    # df['TIME'] = df.index.time
    # df = df.groupby('TIME').agg(
    #     MEAN=(series.name, 'mean'),
    #     SD=(series.name, 'std')
    # )

    if each_month:
        # If aggregated for each month, aggs contains a MultiIndex
        pass
    else:
        # If *not* aggregated for each month, convert to MultiIndex to keep consistent
        aggs = pd.concat({'ALL_MONTHS': aggs}, names=[series.index.name])

    return aggs


def _example_resample_to_monthly_agg_matrix():
    from diive.configs.exampledata import load_exampledata_parquet_long
    df = load_exampledata_parquet_long()
    series = df['Tair_f'].copy()
    import diive as dv
    monthly = dv.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=True)
    print(monthly)


if __name__ == '__main__':
    _example_resample_to_monthly_agg_matrix()
