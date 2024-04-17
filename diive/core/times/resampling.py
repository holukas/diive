from typing import Literal

import pandas as pd
from pandas import Series
from pandas.tseries.frequencies import to_offset

from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import convert_series_timestamp_to_middle
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def resample_series_to_30MIN(series: Series,
                             to_freqstr: Literal['30T', '30min'] = '30min',
                             agg: Literal['mean', 'sum'] = 'mean',
                             mincounts_perc: float = .9,
                             output_timestamp_shows: Literal['middle', 'end'] = 'end') -> Series:
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

    current_freq = to_offset(series.index.freqstr)
    requested_freq = to_offset(to_freqstr)

    # Resampling only 30MIN time resolution
    if not any(chr in to_freqstr for chr in ['30T', '30min']):
        raise NotImplementedError("Error during resampling: Only resampling to 30 minutes (30min) allowed.")

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

    _series = series.copy()

    # Make middle timestamp, for correct resampling
    _series = convert_series_timestamp_to_middle(data=_series)

    print(f"Resampling data from {current_freq.freqstr} "
          f"to {requested_freq.freqstr} frequency ...")

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
    agg_ser = agg_ser[filter_min]

    # Re-assign 30MIN resolution as freq
    agg_ser = agg_ser.asfreq(to_freqstr)

    # Sanitize resampled timestamp index
    if not agg_ser.empty:
        if output_timestamp_shows == 'middle':
            agg_ser = TimestampSanitizer(data=agg_ser).get()
        elif output_timestamp_shows == 'end':
            agg_ser = TimestampSanitizer(data=agg_ser, output_middle_timestamp=False).get()

    return agg_ser
