"""

BASE CLASS FOR QUALITY FLAGS

"""

import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex


class FlagBase():

    def __init__(self, series: Series, flagid: str, levelid: str = None):
        self.series = series
        self._flagid = flagid
        self._levelid = levelid
        self._flagname = self._generate_flagname()

        self._filteredseries = None
        self._flag = None

        print(f"Generating flag {self._flagname} for variable {self.series.name} ...")

    @property
    def flag(self) -> Series:
        """Return flag as Series"""
        if not isinstance(self._flag, Series):
            raise Exception(f'Flag is empty. '
                            f'Solution: run .calc() to create flag for {self.series.name}.')
        return self._flag

    @property
    def filteredseries(self) -> Series:
        """Return data without rejected records"""
        if not isinstance(self._filteredseries, Series):
            raise Exception(f'Filtered data not available. '
                            f'Solution: run .calc() to create flag for {self.series.name}.')
        return self._filteredseries

    def setflag(self, ok: DatetimeIndex, rejected: DatetimeIndex):
        """Set flag with values 0=ok, 2=rejected"""
        self._flag.loc[ok] = 0
        self._flag.loc[rejected] = 2

    def setfiltered(self, rejected: DatetimeIndex):
        """Set rejected values to missing"""
        self._filteredseries.loc[rejected] = np.nan

    def reset(self):
        self._filteredseries = self.series.copy()
        # Generate flag series with NaNs
        self._flag = pd.Series(index=self.series.index, data=np.nan, name=self._flagname)

    def _generate_flagname(self) -> str:
        """Generate standardized name for flag variable"""
        flagname = "FLAG"
        if self._levelid: flagname += f"_L{self._levelid}"
        flagname += f"_{self.series.name}"
        if self._flagid: flagname += f"_{self._flagid}"
        flagname += f"_TEST"
        return flagname
