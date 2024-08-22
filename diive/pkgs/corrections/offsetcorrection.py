from pandas import Series

import diive.core.dfun.frames as frames
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


@ConsoleOutputDecorator()
def remove_relativehumidity_offset(series: Series,
                                   showplot: bool = False) -> Series:
    """Correct relative humidity offset > 100%

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
    _series_corr = series.sub(_offset)

    # Set maximum to 100
    series_corr_max100 = _series_corr.copy()
    still_above_100_locs = series_corr_max100 > 100
    series_corr_max100.loc[still_above_100_locs] = 100
    series_corr_max100.rename("corrected_by_offset_and_max100", inplace=True)

    # Plot
    if showplot:
        from diive.core.plotting.plotfuncs import quickplot
        quickplot([series, _series_exceeds,
                   _daily_mean_above_100, _offset, _series_corr, series_corr_max100],
                  subplots=True,
                  showplot=showplot, hline=100,
                  title=f"Remove RH offset from {outname}")

    # Rename for output
    series_corr_max100.rename(outname, inplace=True)

    return series_corr_max100


@ConsoleOutputDecorator()
def remove_radiation_zero_offset(series: Series,
                                 lat: float,
                                 lon: float,
                                 utc_offset: int,
                                 showplot: bool = False) -> Series:
    """Correct nighttime offset from all radiation data and set nighttime to zero

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
        utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
            The datetime index of the resulting Series will be in this timezone.
        showplot: Show plot

    Returns:
        Corrected series
    """

    outname = series.name
    series.name = "input_data"

    # Detect nighttime
    dnf = DaytimeNighttimeFlag(
        timestamp_index=series.index,
        nighttime_threshold=0.001,
        lat=lat,
        lon=lon,
        utc_offset=utc_offset)
    nighttimeflag = dnf.get_nighttime_flag()
    # daytime = dnf.get_daytime_flag()

    nighttime_ix = nighttimeflag == 1

    # Get series nighttime data
    series_nighttime = series.loc[nighttime_ix]
    nighttime_datetimes = series_nighttime.index

    # Calculate offset as the daily nighttime mean
    _offset = frames.aggregated_as_hires(aggregate_series=series_nighttime,
                                         to_freq='D',
                                         to_agg='mean',
                                         hires_timestamp=series.index,
                                         interpolate_missing_vals=True)


    # from diive.core.plotting.timeseries import TimeSeries
    # TimeSeries(series=_offset).plot()


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
