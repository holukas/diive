"""
NIGHTTIME FLAG
==============
from site location, latitude/longitude

- https://pysolar.org/
- https://pysolar.readthedocs.io/en/latest/#
- https://stackoverflow.com/questions/69766581/pysolar-get-azimuth-function-applied-to-pandas-dataframe
"""

import pandas as pd
from numpy import nan
from pandas import DataFrame, Series
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude_fast

from diive.core.times.times import add_timezone_info


def nighttime_flag_from_latlon(
        lat: float,
        lon: float,
        start: str,
        stop: str,
        freq: str,
        timezone_of_timestamp: str,
        threshold_daytime:float=0
) -> Series:
    """Calculate flag for nighttime based on site location (latitude/longitude)

    Args:
        lat: Latitude of location as float, e.g. 46.583056
        lon: Longitude of location as float, e.g. 9.790639
        start: start date
        stop: end date
        freq: time resolution of resulting dataframe, e.g. '30T' for 30-minute resolution
        timezone_of_timestamp: timezone in the format e.g. 'UTC+01:00' for CET (UTC + 1 hour)
            The datetime index of the resulting Series will be in this timezone.
        threshold_daytime: define as daytime when sun altitude (deg) larger than threshold

    Returns:
        Flag 0/1, where 1=nighttime, 0=daytime

    Example notebook:
        diive/Create Variable/Nighttime Flag From Latitude Longitude.ipynb
        in https://gitlab.ethz.ch/gl-notebooks/general-notebooks

    """

    # Collect data in dataframe
    df = pd.DataFrame()

    # Create timestamp index at requested time resolution and add timezone info
    timestamp_ix = pd.date_range(start=start, end=stop, freq=freq)
    timestamp_ix = add_timezone_info(timezone_of_timestamp=timezone_of_timestamp,
                                     timestamp_index=timestamp_ix)
    df['TIMESTAMP_END'] = timestamp_ix

    # Add UTC timestamp, needed for pysolar
    df['TIMESTAMP_UTC_END'] = df['TIMESTAMP_END'].dt.tz_convert('UTC')

    # Altitude of the sun
    # Calculate the angle between the sun and a plane tangent to the earth for each timestamp
    print(f"Calculating sun altitude in {freq} time resolution ...")
    df['ALTITUDE_SUN_deg'] = get_altitude_fast(lat, lon, df['TIMESTAMP_UTC_END'])
    # df['ALTITUDE_SUN_deg'] = \
    #     df.apply(lambda row: get_altitude(lat, lon, row['TIMESTAMP_UTC_END'].to_pydatetime()), axis=1)

    # Calculate flag
    print("Generating nighttime flag (1=nighttime, 0=daytime) from sun altitude ...")
    df['FLAG_NIGHTTIME_LATLON'] = nan
    df.loc[df['ALTITUDE_SUN_deg'] < threshold_daytime, 'FLAG_NIGHTTIME_LATLON'] = 1
    df.loc[df['ALTITUDE_SUN_deg'] > threshold_daytime, 'FLAG_NIGHTTIME_LATLON'] = 0

    # Remove timezone info from timestamp
    df['TIMESTAMP_END'] = df['TIMESTAMP_END'].dt.tz_localize(None)

    # Set timestamp as index
    df.set_index('TIMESTAMP_END', inplace=True)
    df = df.asfreq(freq)

    return df['FLAG_NIGHTTIME_LATLON']


def potrad_from_latlon(
        lat: float,
        lon: float,
        start: str,
        stop: str,
        freq: str,
        timezone_of_timestamp: str
) -> DataFrame:
    #TODO
    """Calculate potential radiation from latitude/longitude

    Args:
        lat: Latitude of location as float, e.g. 46.583056
        lon: Longitude of location as float, e.g. 9.790639
        timestamp_ix: Timestamp for which potential radiation is calculated
        timezone: 'cet' (Central European Time) or 'utc'

    """
    # Collect data in dataframe
    df = pd.DataFrame()

    # Create timestamp index at requested time resolution and add timezone info
    # TODO timestamp_ix = pd.date_range(start=start, end=stop, freq=freq)
    timestamp_ix = pd.date_range(start='2022-07-01', end='2023-01-01', freq='30T')
    timestamp_ix = add_timezone_info(timezone_of_timestamp=timezone_of_timestamp,
                                     timestamp_index=timestamp_ix)
    df['TIMESTAMP_END'] = timestamp_ix

    # Add UTC timestamp, needed for pysolar
    df['TIMESTAMP_UTC_END'] = df['TIMESTAMP_END'].dt.tz_convert('UTC')

    # Altitude of the sun
    # Calculate the angle between the sun and a plane tangent to the earth for each timestamp
    print(f"Calculating sun altitude in {freq} time resolution ...")
    # df['ALTITUDE_SUN_deg'] = \
    #     df.apply(lambda row: get_altitude(lat, lon, row['TIMESTAMP_UTC_END'].to_pydatetime()), axis=1)
    df['ALTITUDE_SUN_deg'] = get_altitude_fast(lat, lon, df['TIMESTAMP_UTC_END'])

    import matplotlib.pyplot as plt
    df['ALTITUDE_SUN_deg'].plot()
    plt.show()

    df['DNI'] = get_radiation_direct(df['TIMESTAMP_UTC_END'], df['ALTITUDE_SUN_deg'])

    df['DNI'].plot()
    plt.show()
    df['DNI'].groupby(df['TIMESTAMP_UTC_END'].dt.month).mean().plot()
    plt.show()

    import numpy as np
    df['POTENTIAL_RADIATION'] = df['DNI'] * np.cos(np.deg2rad(df['ALTITUDE_SUN_deg']))

    df['POTENTIAL_RADIATION'].plot()
    plt.show()

    # Calculate flag
    print("Generating nighttime flag (1=nighttime, 0=daytime) from sun altitude ...")
    df['FLAG_NIGHTTIME_LATLON'] = nan
    df.loc[df['ALTITUDE_SUN_deg'] < 0, 'FLAG_NIGHTTIME_LATLON'] = 1
    df.loc[df['ALTITUDE_SUN_deg'] > 0, 'FLAG_NIGHTTIME_LATLON'] = 0

    # Remove timezone info from timestamp
    df['TIMESTAMP_END'] = df['TIMESTAMP_END'].dt.tz_localize(None)

    # Set timestamp as index
    df.set_index('TIMESTAMP_END', inplace=True)
    df = df.asfreq(freq)
    return df



    # # Potential radiation
    # # https://pysolar.readthedocs.io/en/latest/#
    # # https://stackoverflow.com/questions/69766581/pysolar-get-azimuth-function-applied-to-pandas-dataframe
    # # CH-AWS: 46.583056, 9.790639
    #
    # # # Calculate the azimuth of the sun
    # # print(get_azimuth(lat, lon, date))
    # # radiation.get_radiation_direct(date, altitude_deg)
    # # CET = UTC + 1
    # # _series = series.resample('30T').mean()
    #
    # df = pd.DataFrame()
    #
    # # Create column for UTC
    # if timezone == 'cet':
    #     df['cet'] = timestamp_ix  # Timezone is Central European Time (always)
    #     df['utc'] = df['cet'].sub(datetime.timedelta(hours=1))  # Calculate UTC
    # elif timezone == 'utc':
    #     df['utc'] = timestamp_ix
    #
    # # Defince UTC column as UTC timezone
    # df['utc'] = pd.to_datetime(df['utc'], utc=True)
    #
    # # Calculate the angle between the sun and a plane tangent to the earth for each timestamp
    # df['altitude_deg'] = df.apply(lambda row: get_altitude(lat, lon, row['utc'].to_pydatetime()), axis=1)
    #
    # # Set timezone of input data as index
    # if timezone == 'cet':
    #     df.set_index('cet', inplace=True)
    # elif timezone == 'utc':
    #     df.set_index('utc', inplace=True)
    #
    # # Calculate potential radiation for each timestamp
    # df['pot_rad'] = df.apply(
    #     lambda row: radiation.get_radiation_direct(row['utc'].to_pydatetime(), row['altitude_deg']), axis=1)
    #
    # return df['pot_rad']


def example():
    # Example for CH-DAV
    nighttime_flag = nighttime_flag_from_latlon(
        lat=46.815333,
        lon=9.855972,
        start='2021-01-01 00:30:00',
        stop='2022-12-31 00:00:00',
        freq='30T',
        timezone_of_timestamp='UTC+01:00')
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=nighttime_flag).show()


if __name__ == '__main__':
    example()
