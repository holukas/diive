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
from pandas import Series, DataFrame, DatetimeIndex

from diive.pkgs.createvar.potentialradiation import potrad


class DaytimeNighttimeFlag:
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
        daytime, nighttime = daytime_nighttime_flag_from_swinpot(
            swinpot=self.swinpot, nighttime_threshold=self.nighttime_threshold)
        return daytime, nighttime


def daytime_nighttime_flag_from_swinpot(swinpot: Series,
                                        nighttime_threshold: float = 50,
                                        daytime_col: str = 'DAYTIME',
                                        nighttime_col: str = 'NIGHTTIME') -> tuple[Series, Series]:
    """
    Create flags to identify daytime and nighttime data

    Args:
        swinpot: Potential short-wave incoming radiation (W m-2)
        nighttime_threshold: Threshold below which data are flagged as nighttime (W m-2)
        daytime_col: Output variable name of the daytime flag
        nighttime_col: Output variable name of the nighttime flag

    Returns:
        Flags as two separate Series:
            *daytime* with flags 1=daytime, 0=not daytime
            *nighttime* with flags 1=nighttime, 0=not nighttime
    """
    daytime = pd.Series(index=swinpot.index, data=np.nan, name=daytime_col)
    daytime.loc[swinpot >= nighttime_threshold] = 1  # Yes, it is daytime
    daytime.loc[swinpot < nighttime_threshold] = 0  # No, it is not daytime
    nighttime = pd.Series(index=swinpot.index, data=np.nan, name=nighttime_col)
    nighttime.loc[swinpot >= nighttime_threshold] = 0
    nighttime.loc[swinpot < nighttime_threshold] = 1
    return daytime, nighttime


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()

    dnf = DaytimeNighttimeFlag(
        timestamp_index=df.index,
        nighttime_threshold=50,
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
