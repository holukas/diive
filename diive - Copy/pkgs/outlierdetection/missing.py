from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase

from diive.core.utils.prints import ConsoleOutputDecorator
@ConsoleOutputDecorator()
class MissingValues(FlagBase):
    """
    Generate flag that indicates missing records in data
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
    flagid = 'MISSING'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)

    def calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""
        rejected = self.series.isnull()  # Yields boolean series
        if rejected.sum() == 0:  # No missing values
            ok = self.series.index
            rejected = None
        else:
            ok = ~rejected
            rejected = rejected[rejected].index
            ok = ok[ok].index
        return ok, rejected


def example():
    pass


if __name__ == '__main__':
    example()
