import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


def restrict_application(flag: Series, flagname: str, application_dates: list or None = None,
                         verbose: bool = True, fill_value: float = np.nan):
    """Apply flag during certain time periods only.

    Args:
        flag: Pandas Series containing the flag values.
        flagname: Name of the flag.
        application_dates: List of dates or date ranges (as lists) when the flag should be applied.
        verbose: Whether to print information about the application dates.
        fill_value: Value to fill in for periods outside the application dates.

    Returns:
        Pandas Series containing the restricted flag.
    """
    infotxt = ""
    orig_flag = flag.copy()
    restricted_flag = pd.Series(index=orig_flag.index, data=fill_value)
    if verbose:
        print(f"{flagname}: will be applied on the following dates only: {application_dates}")

    for date in application_dates:
        if isinstance(date, str):
            # Neat solution: even though here only data for a single datetime
            # is removed, the >= and <= comparators are used to avoid an error
            # in case the datetime is not found in the flag.index
            dates = (restricted_flag.index >= date) & (restricted_flag.index <= date)
            restricted_flag.loc[dates] = orig_flag.loc[dates].copy()
        elif isinstance(date, list):
            dates = (restricted_flag.index >= date[0]) & (restricted_flag.index <= date[1])
            restricted_flag.loc[dates] = orig_flag.loc[dates].copy()

    return restricted_flag


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

    def __init__(self, series: Series, idstr: str = None, verbose: bool = False):
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.verbose = verbose

    def calc(self, repeat: bool = False):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=False)
        # if self.showplot:
        #     self.defaultplot(n_iterations=n_iterations)

        if self.verbose:
            print(f"MISSING VALUES TEST: Generated new flag variable {self.overall_flag.name}, "
                  f"newly calculated from variable {self.series.name},"
                  f"with flag 0 (good values) where {self.series.name} is available, "
                  f"flag 2 (bad values) where {self.series.name} is missing.")

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""
        # Maybe check this alternative method:
        # missing_values_flag = np.where(df[var].isnull(), 2, 0)
        # missing_values_flag = Series(missing_values_flag, index=df.index, name=flagname_out)
        rejected = self.series.isnull()  # Yields boolean series
        if rejected.sum() == 0:  # No missing values
            ok = self.series.index
            rejected = None
        else:
            ok = ~rejected
            rejected = rejected[rejected].index
            ok = ok[ok].index

        # No outliers are detected in this test, only already missing values are flagged
        n_outliers = 0

        return ok, rejected, n_outliers
