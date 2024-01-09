"""
OUTLIER DETECTION: MANUAL REMOVAL
=================================

This module is part of the diive library:
https://github.com/holukas/diive

"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.repeater import repeater


@ConsoleOutputDecorator()
@repeater  # Repeater called for consistency with other methods, ManualRemoval does not require iterations
class ManualRemoval(FlagBase):
    """Generate flag for data points that should be removed."""

    flagid = 'OUTLIER_MANUAL'

    def __init__(self,
                 series: Series,
                 remove_dates: list,
                 showplot: bool = False,
                 verbose: bool = False,
                 idstr: str = None,
                 repeat: bool = False):
        """

        Args:
            series: Time series in which outliers are identified.
            remove_dates: list, can be given as a mix of strings and lists that
                contain the date(times) of records that should be removed.
                Example:
                    * remove_dates=['2022-06-30 23:58:30', ['2022-06-05 00:00:30', '2022-06-07 14:30:00']]*
                    will remove the record for '2022-06-30 23:58:30' and all records between
                    '2022-06-05 00:00:30' (inclusive) and '2022-06-07 14:30:00' (inclusive).
                    * This also works when providing only the date, e.g.
                    removed_dates=['2006-05-01', '2006-07-18'] will remove all data points between
                    2006-05-01 and 2006-07-18.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.
            idstr: Identifier, added as suffix to output variable names.
            repeat: Repeat until no more outliers can be found.

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.remove_dates = remove_dates
        self.showplot = showplot
        self.verbose = verbose
        self.repeat = repeat

    def _calc(self):
        """Calculate flag."""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        flag = pd.Series(index=self.series.index, data=np.nan)

        # Location of rejected records
        for date in self.remove_dates:
            if isinstance(date, str):
                # Neat solution: even though here only data for a single datetime
                # is removed, the >= and <= comparators are used to avoid an error
                # in case the datetime is not found in the flag.index
                date = (flag.index >= date) & (flag.index <= date)
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

        if self.showplot:
            self.plot(ok=ok, rejected=rejected)

        return ok, rejected


def example():
    pass


if __name__ == '__main__':
    example()
