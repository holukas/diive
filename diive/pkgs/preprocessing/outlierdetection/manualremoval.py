"""
MANUAL REMOVAL: EXPLICIT DATA FLAGGING
=======================================

Manually flag specific records or date ranges as outliers.

Part of the diive library: https://github.com/holukas/diive
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
        self.remove_dates = remove_dates
        self.showplot = showplot
        self.verbose = verbose

    def calc(self):
        """Calculate overall flag for manually removed data points."""

        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=False)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration: int) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Mark manually specified dates as outliers.

        Args:
            iteration: Current iteration number

        Returns:
            (ok_indices, rejected_indices, n_outliers) where:
                - ok_indices: DatetimeIndex of records to keep
                - rejected_indices: DatetimeIndex of manually removed records
                - n_outliers: Total number of removed records
        """

        flag = pd.Series(index=self.filteredseries.index, data=np.nan)

        for date_spec in self.remove_dates:
            if isinstance(date_spec, str):
                date_mask = (flag.index >= date_spec) & (flag.index <= date_spec)
                flag.loc[date_mask] = 2
            elif isinstance(date_spec, list):
                date_mask = (flag.index >= date_spec[0]) & (flag.index <= date_spec[1])
                flag.loc[date_mask] = 2

        rejected = flag[flag == 2].index
        n_outliers = len(rejected)
        ok = flag.index.difference(rejected)

        if self.verbose:
            print(f"ITERATION#{iteration}: Manually removed {n_outliers} values")

        return ok, rejected, n_outliers
