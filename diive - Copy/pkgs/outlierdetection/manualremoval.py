import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class ManualRemoval(FlagBase):
    """
    Generate flag for data points that should be removed
    ...

    Methods:
        calc(): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    """
    flagid = 'OUTLIER_MANUAL'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.verbose = False

    def calc(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """
        Calculate flag

        Args:
            remove_dates: list, can be given as a mix of strings and lists that
                contain the date(times) of records that should be removed
                Example:
                    *remove_dates=['2022-06-30 23:58:30', ['2022-06-05 00:00:30', '2022-06-07 14:30:00']]*
                    will remove the record for '2022-06-30 23:58:30' and all records between
                    '2022-06-05 00:00:30' (inclusive) and '2022-06-07 14:30:00' (inclusive).

        """

        self.showplot = showplot
        self.verbose = verbose
        self.reset()
        ok, rejected = self._flagtests(remove_dates=remove_dates)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self, remove_dates) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        flag = pd.Series(index=self.series.index, data=np.nan)

        # Location of rejected records
        for date in remove_dates:
            if isinstance(date, str):
                flag.loc[date] = 2
            elif isinstance(date, list):
                dates = (flag.index >= date[0]) & (flag.index <= date[1])
                flag.loc[dates] = 2

        rejected = flag == 2
        rejected = rejected[rejected]

        # Index of rejected records
        rejected = rejected.index

        # All records that were not rejected are OK
        ok = flag.index.difference(rejected)

        if self.showplot: self.plot(ok=ok, rejected=rejected)

        return ok, rejected


def example():
    pass


if __name__ == '__main__':
    example()
