"""
NIGHTTIME FLAG
==============
from site location, latitude/longitude

- https://pysolar.org/
- https://pysolar.readthedocs.io/en/latest/#
- https://stackoverflow.com/questions/69766581/pysolar-get-azimuth-function-applied-to-pandas-dataframe
"""

import numpy as np
import pandas as pd
from numpy import nan
from pandas import Series, DataFrame, DatetimeIndex
from pysolar.solar import get_altitude_fast

from diive.core.times.times import add_timezone_info
from diive.pkgs.createvar.potentialradiation import potrad


class DaytimeNighttimeFlag:  # TODO HIER WEITER
    """
    Create flags to identify daytime and nighttime data
    """

    swinpot_col = 'SW_IN_POT'
    daytime_col = 'DAYTIME'
    nighttime_col = 'NIGHTTIME'

    def __init__(self,
                 timestamp_index: DatetimeIndex,
                 utc_offset: int,
                 lat: float,
                 lon: float,
                 nighttime_threshold: float = 50):
        """

        Args:
            timestamp_index: Time series index, flags and potential radiation
                are calculated using this index
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
            lat: Latitude
            lon: Longitude
            nighttime_threshold: Threshold for potential radiation below which data
                are flagged as nighttime (W m-2)

        """

        self.timestamp_index = timestamp_index
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.lat = lat
        self.lon = lon

        self.swinpot = None
        self.daytime = None
        self.nighttime = None
        self._df = None

        self._run()

    @property
    def df(self) -> DataFrame:
        """Get dataframe with potential radiation and daytime/nighttime flags"""
        if not isinstance(self._df, DataFrame):
            raise Exception('Data empty.')
        return self._df

    def get_daytime_flag(self) -> Series:
        """Return daytime flag where 1=daytime, 0=nighttime"""
        return self.df[self.daytime_col]

    def get_nighttime_flag(self) -> Series:
        """Return nighttime flag where 0=daytime, 1=nighttime"""
        return self.df[self.nighttime_col]

    def get_swinpot(self) -> Series:
        """Return potential radiation"""
        return self.df[self.swinpot_col]

    def _run(self):
        self._calc_swin_pot()
        self._calc_flags()
        self._assemble()

    def _assemble(self):
        frame = {
            self.swinpot_col: self.swinpot,
            self.daytime_col: self.daytime,
            self.nighttime_col: self.nighttime
        }
        self._df = DataFrame.from_dict(frame)

    def _calc_swin_pot(self):
        """Calculate potential radiation from latitude and longitude"""
        self.swinpot = potrad(timestamp_index=self.timestamp_index,
                              lat=self.lat,
                              lon=self.lon,
                              utc_offset=self.utc_offset)

    def _calc_flags(self):
        self.daytime, self.nighttime = self._daytime_nighttime_flag_from_swinpot()

    def _daytime_nighttime_flag_from_swinpot(self) -> tuple[Series, Series]:
        daytime = pd.Series(index=self.swinpot.index, data=np.nan, name=self.daytime_col)
        daytime.loc[self.swinpot >= self.nighttime_threshold] = 1  # Yes, it is daytime
        daytime.loc[self.swinpot < self.nighttime_threshold] = 0  # No, it is not daytime
        nighttime = pd.Series(index=self.swinpot.index, data=np.nan, name=self.nighttime_col)
        nighttime.loc[self.swinpot >= self.nighttime_threshold] = 0
        nighttime.loc[self.swinpot < self.nighttime_threshold] = 1
        return daytime, nighttime


def daytime_nighttime_flag_from_swinpot(swinpot: Series,
                                        nighttime_threshold: float = 50) -> tuple[Series, Series]:
    """
    Create flags to identify daytime and nighttime data

    Args:
        swinpot: Potential short-wave incoming radiation (W m-2)
        nighttime_threshold: Threshold below which data are flagged as nighttime (W m-2)

    Returns:
        Flags as two separate Series:
            *daytime* with flags 1=daytime, 0=not daytime
            *nighttime* with flags 1=nighttime, 0=not nighttime
    """
    daytime = pd.Series(index=swinpot.index, data=np.nan, name='DAYTIME')
    daytime.loc[swinpot >= nighttime_threshold] = 1  # Yes, it is daytime
    daytime.loc[swinpot < nighttime_threshold] = 0  # No, it is not daytime
    nighttime = pd.Series(index=swinpot.index, data=np.nan, name='NIGHTTIME')
    nighttime.loc[swinpot >= nighttime_threshold] = 0
    nighttime.loc[swinpot < nighttime_threshold] = 1
    return daytime, nighttime


def nighttime_flag_from_latlon(
        lat: float,
        lon: float,
        start: str,
        stop: str,
        freq: str,
        timezone_of_timestamp: str,
        threshold_daytime: float = 0
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


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()

    dnf = DaytimeNighttimeFlag(
        timestamp_index=df.index,
        nighttime_threshold=1,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1)

    # dnf.get_daytime_flag()
    # dnf.get_nighttime_flag()
    # dnf.get_swinpot()

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=dnf.df['DAYTIME']).show()
    HeatmapDateTime(series=dnf.df['NIGHTTIME']).show()
    HeatmapDateTime(series=dnf.df['SW_IN_POT']).show()


if __name__ == '__main__':
    example()
