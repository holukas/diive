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


@ConsoleOutputDecorator()
class ManualRemoval(FlagBase):
    """Generate flag for data points that should be removed."""

    flagid = 'OUTLIER_MANUAL'

    def __init__(self,
                 series: Series,
                 remove_dates: list,
                 showplot: bool = False,
                 verbose: bool = False,
                 idstr: str = None):
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

    def calc(self, repeat: bool = False):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=False)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        flag = pd.Series(index=self.filteredseries.index, data=np.nan)

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

        n_outliers = len(rejected)

        # Index of rejected records
        rejected = rejected.index

        # All records that were not rejected are OK
        ok = flag.index.difference(rejected)

        return ok, rejected, n_outliers


def example():
    pass


if __name__ == '__main__':
    example()
