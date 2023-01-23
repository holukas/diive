from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class AbsoluteLimits(FlagBase):
    """
    Generate flag that indicates if values in data are outside
    the specified range, defined by providing min, max in method
    ...

    Methods:
        calc(self, min: float, max: float): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    """
    flagid = 'OUTLIER_ABSLIM'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)

    def calc(self, min: float, max: float):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests(min, max)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self, min, max) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""
        ok = (self.series >= min) | (self.series <= max)
        ok = ok[ok].index
        rejected = (self.series < min) | (self.series > max)
        rejected = rejected[rejected].index
        return ok, rejected


# @ConsoleOutputDecorator()
# def absolute_limits(series: Series, min: int, max: int) -> Series:
#     # flag_name = f"QCF_OUTLIER_ABSLIM_{series.name}"
#     flag = pd.Series(index=series.index, data=False)
#     flag.loc[series < min] = True
#     flag.loc[series > max] = True
#     flag.name = flag_name
#     return flag

def example():
    import numpy as np
    import pandas as pd
    np.random.seed(100)
    rows = 1000
    data = np.random.rand(rows) * 100  # Random numbers b/w 0 and 100
    tidx = pd.date_range('2019-01-01 00:30:00', periods=rows, freq='30T')
    series = pd.Series(data, index=tidx, name='TESTDATA')

    al = AbsoluteLimits(series=series, levelid='99')
    al.calc(min=16, max=84)

    print(series.describe())
    filteredseries = al.filteredseries
    print(filteredseries.describe())
    flag = al.flag
    print(flag.describe())


if __name__ == '__main__':
    example()
