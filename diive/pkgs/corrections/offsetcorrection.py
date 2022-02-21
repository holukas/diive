from pathlib import Path

import pandas as pd
from pandas import Series

import diive.core.dfun.frames as frames
# from diive.common.dfun.frames import resample_df
from diive.pkgs.createvar.potentialradiation import potrad_from_latlon


def remove_relativehumidity_offset(series: Series,
                                   show: bool = False,
                                   saveplot: str or Path = None) -> Series:
    """Remove relative humidity offset

    Works for relative humidity data where maximum values should not exceed 100%.

    Args:
        series: Data for relative humidity variable that is corrected
        show: Show plot

    Returns:
    Corrected series
    """

    print(f"Removing RH offset from {series.name} ...")
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
    series_corr.rename(outname, inplace=True)

    # Plot
    if saveplot:
        from diive.core.plotting.plotfuncs import quickplot_df
        quickplot_df([series, _series_exceeds, _daily_mean_above_100, _offset, series_corr], subplots=False,
                     saveplot=saveplot, hline=100, title=f"Remove RH Offset From {series.name}")

    return series_corr


def remove_radiation_zero_offset(_series: Series,
                                 lat: float,
                                 lon: float,
                                 show: bool = False,
                                 saveplot: str or Path = None) -> Series:
    """Remove nighttime offset from radiation data

    Works for radiation data where radiation should be zero during nighttime.

    Potential radiation is calculated and used to detect nighttime data. Then,
    for each radiation variable, the nighttime mean is calculated for each
    day in the dataset. This mean is the offset by which daytime and nighttime
    radiation data are corrected. Finally, after the application of the offset,
    nighttime data are set to zero.

    Args:
        _series: Data for radiation variable that is corrected
        lat: Latitude of the location where radiation data were recorded
        lon: Longitude of the location where radiation data were recorded
        show: Show plot

    Returns:
    Corrected series
    """

    print(f"Removing radiation zero-offset from {_series.name} ...")
    outname = _series.name
    _series.name = "input_data"

    # Calculate potential radiation for lower time resolution (30MIN)
    lower_res, _ = frames.resample_df(df=pd.DataFrame(_series), freq_str='30T', agg='mean', mincounts_perc=.9,
                                      to_freq='T')
    _potential_radiation = potrad_from_latlon(lat=lat, lon=lon, timestamp_ix=lower_res.index)
    _potential_radiation = _potential_radiation.reindex(_series.index, method='nearest')  # Reindex to hires timestamp
    _potential_radiation.rename("potential_radiation", inplace=True)

    # Detect series nighttime data from potential radiation
    series_nighttime = _series.loc[_potential_radiation == 0]
    nighttime_ix = series_nighttime.index

    # Calculate offset as the daily nighttime mean
    _offset = frames.aggregated_as_hires(aggregate_series=series_nighttime,
                                         to_freq='D',
                                         to_agg='mean',
                                         hires_timestamp=_series.index,
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
    _series_corr = _series.sub(_offset)
    _series_corr.rename("corrected_by_offset", inplace=True)

    # Set nighttime radiation to zero (based on potential radiation)
    series_corr_settozero = _series_corr.copy()
    series_corr_settozero.loc[nighttime_ix] = 0

    # # Set still remaining negative values to zero
    # series_corr_settozero.loc[series_corr_settozero < 0] = 0

    series_corr_settozero.rename(outname, inplace=True)

    # Plot
    if saveplot:
        from diive.core.plotting.plotfuncs import quickplot_df
        quickplot_df([_series, _potential_radiation, _series_corr,
                      series_corr_settozero, _offset],
                     subplots=False,
                     saveplot=saveplot,
                     title=f"Removing radiation zero-offset from {_series.name}")
        # Separate plot showing the offset
        quickplot_df([_offset],
                     subplots=False,
                     saveplot=saveplot,
                     title=f"Used offset values from removing radiation zero-offset from {_series.name}")

    return series_corr_settozero


if __name__ == '__main__':
    pass
