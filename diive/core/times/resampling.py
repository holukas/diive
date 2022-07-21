import pandas as pd
from pandas import DataFrame,Series
from diive.core.times.times import convert_series_timestamp_to_middle, sanitize_timestamp_index
from pandas.tseries.frequencies import to_offset

def resample_series_to_30MIN(series: Series,
                             to_freqstr: str = '30T',
                             agg: str = 'mean',
                             mincounts_perc: float = .9):
    """Downsample data to 30-minute time resolution

    Input data must have timestamp showing the END of the time period.
    Before resampling, the timestamp is converted to show the MIDDLE
    of the time period. After resampling, the timestamp shows again the
    END of the time period.

    Using the selected aggregation method and while also considering the
    minimum required values in the aggregation time window.

    Note regarding .resample:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        closed : {‘right’, ‘left’}, default None
            Which side of bin interval is closed. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

        label : {‘right’, ‘left’}, default None
            Which bin edge label to label bucket with. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

    By default, for weekly aggregation the first day of the week in pandas is Sunday, but diive uses Monday.

    https://stackoverflow.com/questions/48340463/how-to-understand-closed-and-label-arguments-in-pandas-resample-method
        closed='right' =>  ( 3:00, 6:00 ]  or  3:00 <  x <= 6:00
        closed='left'  =>  [ 3:00, 6:00 )  or  3:00 <= x <  6:00

    """

    # Resampling only 30MIN time resolution
    if not any(chr in to_freqstr for chr in ['30T']):
        raise NotImplementedError("Error during resampling: Only resampling to 30 minutes (30T) allowed.")

    # Timestamp must be regular
    if not series.index.freq:
        raise NotImplementedError("Error during resampling: Irregular timestamps are not supported.")

    # Requested frequency must be different from data freq
    # 'to_offset' allows comparison of frequencies (larger, smaller, equal)
    if to_offset(series.index.freqstr) == to_offset(to_freqstr):
        return series

    # Requested frequency must be larger than data freq
    if to_offset(series.index.freqstr) > to_offset(to_freqstr):
        raise NotImplementedError(
            f"Error during resampling: "
            f"Upsampling not allowed. "
            f"Target frequency {to_freqstr} must be lower time resolution than "
            f"source frequency {series.index.freqstr}.")

    _series = series.copy()

    print(f"Resampling data from {series.index.freqstr} to {to_freqstr} frequency ...")

    # Make middle timestamp, for correct resampling
    _series = convert_series_timestamp_to_middle(series=_series)

    # Check maximum number of counts per aggregation interval
    # Needed to calculate the required minimum number of counts from
    # 'mincounts_perc', which is a relative threshold.
    maxcounts = pd.Series(index=_series.index, data=1)  # Dummy series of 1s
    maxcounts = maxcounts.resample(to_freqstr, label='right').count().max()
    # maxcounts = maxcounts.resample(to_freqstr, label=label, closed=closed).count().max()
    mincounts = int(maxcounts * mincounts_perc)

    # Aggregation
    resampled_df = _series.resample(to_freqstr, label='right')  # default: closed='left'
    # resampled_df = _series.resample(to_freqstr, label=label, closed=closed)
    agg_counts_df = resampled_df.count()  # Count aggregated values, always needed
    agg_df = resampled_df.agg(agg)

    # Timestamp index shows end after resampling b/c label='right'
    agg_counts_df.index.name = 'TIMESTAMP_END'
    agg_df.index.name = 'TIMESTAMP_END'

    # Keep aggregates with enough values
    filter_min = agg_counts_df >= mincounts
    agg_df = agg_df[filter_min]

    # Sanitize resampled timestamp index
    agg_df = sanitize_timestamp_index(data=agg_df, freq='30T')

    # # Insert additional timestamps
    # timestamp_freq = agg_df.index.freq
    # timedelta = pd.to_timedelta(timestamp_freq)
    # agg_df['TIMESTAMP_END'] = agg_df.index + pd.Timedelta(timedelta)
    # agg_df['TIMESTAMP_MID'] = agg_df.index + pd.Timedelta(timedelta / 2)

    # print(agg_df)
    # # # TIMESTAMP CONVENTION
    # # # --------------------
    # agg_df, timestamp_info_df = timestamp_convention(df=agg_df,
    #                                                  timestamp_shows_start=timestamp_shows_start,
    #                                                  out_timestamp_convention='Middle of Record')
    # agg_df.index = pd.to_datetime(agg_df.index)

    return agg_df
