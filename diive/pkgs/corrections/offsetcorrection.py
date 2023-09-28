from pandas import Series

import diive.core.dfun.frames as frames
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import nighttime_flag_from_latlon


@ConsoleOutputDecorator()
def remove_relativehumidity_offset(series: Series,
                                   showplot: bool = False) -> Series:
    """Remove relative humidity offset

    Works for relative humidity data where maximum values should not exceed 100%.

    Args:
        series: Data for relative humidity variable that is corrected
        showplot: Show plot

    Returns:
        Corrected series
    """

    # print(f"Removing RH offset from {series.name} ...")
    outname = series.name
    series.name = "input_data"

    # Detect series data that exceeds 100% relative humidity
    _series_exceeds = series.loc[series > 100]
    _series_exceeds.rename("input_data_exceeds_100", inplace=True)
    exceeds_ix = _series_exceeds.index

    # Calculate daily mean of values > 100
    _daily_mean_above_100 = frames.aggregated_as_hires(aggregate_series=_series_exceeds,
                                                       to_freq='D',
                                                       to_agg='mean',
                                                       hires_timestamp=series.index,
                                                       interpolate_missing_vals=True)

    # Calculate and gap-fill offset values
    _offset = _daily_mean_above_100.sub(100)  # Offset is the difference to 100
    if _offset.dropna().empty:
        _offset.loc[:] = 0  # Set offset to zero
    _offset = _offset.interpolate().ffill().bfill()
    # _offset = _offset.interpolate().ffill().bfill()
    _offset.rename("offset", inplace=True)

    # Subtract offset from relative humidity (RH) column (series - offset)
    # Offset examples assuming measured RH is 130:
    #   In case of positive offset +30:
    #       (RH must not be larger than 100% but 130% were measured)
    #       130 - (+30) = 100
    # Corrected RH is most likely not *exactly* 100%, but closer to it.
    series_corr = series.sub(_offset)

    # Plot
    if showplot:
        from diive.core.plotting.plotfuncs import quickplot
        quickplot([series, _series_exceeds,
                   _daily_mean_above_100, _offset, series_corr],
                  subplots=True,
                  showplot=showplot, hline=100,
                  title=f"Remove RH offset from {outname}")

    # Rename for output
    series_corr.rename(outname, inplace=True)

    return series_corr


@ConsoleOutputDecorator()
def remove_radiation_zero_offset(series: Series,
                                 lat: float,
                                 lon: float,
                                 timezone_of_timestamp: str,
                                 showplot: bool = False) -> Series:
    """Remove nighttime offset from all radiation data and set nighttime to zero

    Works for radiation data where radiation should be zero during nighttime.

    Nighttime flag is calculated and used to detect nighttime data. Then,
    for each radiation variable, the nighttime radiation mean is calculated for
    each day in the dataset. This mean is the offset by which daytime and nighttime
    radiation data are corrected. Finally, after the application of the offset,
    nighttime data are set to zero.

    Args:
        series: Data for radiation variable that is corrected
        lat: Latitude of the location where radiation data were recorded
        lon: Longitude of the location where radiation data were recorded
        timezone_of_timestamp: timezone in the format e.g. 'UTC+01:00'
            for CET (UTC + 1 hour), needed for nighttime detection via
            pysolar which works with UTC
        showplot: Show plot

    Returns:
        Corrected series
    """

    outname = series.name
    series.name = "input_data"

    # Calculate nighttime flag from sun position (angle) for 10MIN time resolution
    nighttime_flag = nighttime_flag_from_latlon(
        lat=lat, lon=lon,
        start=str(series.index[0]), stop=str(series.index[-1]),
        freq='10T', timezone_of_timestamp=timezone_of_timestamp)

    # Reindex to hires timestamp
    nighttime_flag_in_hires = nighttime_flag.reindex(series.index, method='nearest')
    nighttime_ix = nighttime_flag_in_hires == 1
    # nighttime_flag.rename("nighttime_flag", inplace=True)

    # Get series nighttime data
    series_nighttime = series.loc[nighttime_ix]
    nighttime_datetimes = series_nighttime.index

    # Calculate offset as the daily nighttime mean
    _offset = frames.aggregated_as_hires(aggregate_series=series_nighttime,
                                         to_freq='D',
                                         to_agg='mean',
                                         hires_timestamp=series.index,
                                         interpolate_missing_vals=True)

    # Gap-fill offset values
    _offset = _offset.fillna(_offset.median())
    # offset = offset.interpolate().ffill().bfill()
    _offset.rename("offset", inplace=True)

    # Subtract offset from radiation column (rad_col - offset)
    # Offset examples assuming measured radiation is 120:
    #   In case of negative offset -8:
    #       (measured radiation should be zero but -8 was measured)
    #       120 - (-8) = 128
    #   In case of positive offset +8:
    #       (measured radiation should be zero but +8 was measured)
    #       120 - (+8) = 112
    _series_corr = series.sub(_offset)
    _series_corr.rename("corrected_by_offset", inplace=True)

    # Set nighttime radiation to zero (based on potential radiation)
    series_corr_settozero = _series_corr.copy()
    series_corr_settozero.loc[nighttime_ix] = 0
    series_corr_settozero.rename("corrected_by_offset_and_night_settozero", inplace=True)

    # Set still remaining negative values to zero
    series_corr_settozero.loc[series_corr_settozero < 0] = 0

    # Plot
    if showplot:
        from diive.core.plotting.plotfuncs import quickplot
        quickplot([series, _series_corr,
                   series_corr_settozero, _offset],
                  subplots=True,
                  showplot=showplot,
                  title=f"Removing radiation zero-offset from {outname}")

    # Rename for output
    series_corr_settozero.rename(outname, inplace=True)

    return series_corr_settozero
