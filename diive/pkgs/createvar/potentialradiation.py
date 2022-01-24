import datetime

import pandas as pd
from pandas import DataFrame, DatetimeIndex
from pysolar import radiation
from pysolar.solar import get_altitude
from diive.common.plotting.plotfuncs import quickplot_df

def potrad_from_latlon(
        lat: float,
        lon: float,
        timestamp_ix: DatetimeIndex,
        timezone: str = 'cet'
) -> DataFrame:
    """Calculate potential radiation from latitude/longitude

    Args:
        lat: Latitude of location as float, e.g. 46.583056
        lon: Longitude of location as float, e.g. 9.790639
        timestamp_ix: Timestamp for which potential radiation is calculated
        timezone: 'cet' (Central European Time) or 'utc'

    """
    # Potential radiation
    # https://pysolar.readthedocs.io/en/latest/#
    # https://stackoverflow.com/questions/69766581/pysolar-get-azimuth-function-applied-to-pandas-dataframe
    # CH-AWS: 46.583056, 9.790639

    # # Calculate the azimuth of the sun
    # print(get_azimuth(lat, lon, date))
    # radiation.get_radiation_direct(date, altitude_deg)
    # CET = UTC + 1
    # _series = series.resample('30T').mean()

    df = pd.DataFrame()

    # Create column for UTC
    if timezone == 'cet':
        df['cet'] = timestamp_ix  # Timezone is Central European Time (always)
        df['utc'] = df['cet'].sub(datetime.timedelta(hours=1))  # Calculate UTC
    elif timezone == 'utc':
        df['utc'] = timestamp_ix

    # Defince UTC column as UTC timezone
    df['utc'] = pd.to_datetime(df['utc'], utc=True)

    # Calculate the angle between the sun and a plane tangent to the earth for each timestamp
    df['altitude_deg'] = df.apply(lambda row: get_altitude(lat, lon, row['utc'].to_pydatetime()), axis=1)

    # Set timezone of input data as index
    if timezone == 'cet':
        df.set_index('cet', inplace=True)
    elif timezone == 'utc':
        df.set_index('utc', inplace=True)

    # Calculate potential radiation for each timestamp
    df['pot_rad'] = df.apply(
        lambda row: radiation.get_radiation_direct(row['utc'].to_pydatetime(), row['altitude_deg']), axis=1)

    return df['pot_rad']
