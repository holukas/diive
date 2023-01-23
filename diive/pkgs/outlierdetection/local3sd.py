import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DatetimeIndex
from pandas import Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class LocalSD(FlagBase):
    """
    Identify outliers based on the local standard deviation
    ...

    Methods:
        calc(): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/

    """
    flagid = 'OUTLIER_LOCALSD'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.verbose = False

    def calc(self, n_sd: float = 7, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        self.showplot = showplot
        self.verbose = verbose
        self.reset()
        ok, rejected = self._flagtests(n_sd=n_sd, verbose=verbose, showplot=showplot)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self, n_sd, verbose, showplot) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Working data
        s = self.series.copy()
        s = s.dropna()

        winsize = int(len(s) / 20)
        mean = s.rolling(window=winsize, center=True, min_periods=3).median()
        # sd = s.std()
        sd = s.rolling(window=winsize, center=True, min_periods=3).std()
        upper_limit = mean + (sd*n_sd)
        # upper_limit = mean + sd.multiply(n_sd)
        lower_limit = mean - (sd*n_sd)
        # lower_limit = mean - sd.multiply(n_sd)

        _d = pd.concat([s, mean, upper_limit, lower_limit], axis=1)
        _d.plot()
        plt.show()

        ok = (s < upper_limit) & (s > lower_limit)
        ok = ok[ok].index
        rejected = (s > upper_limit) | (s < lower_limit)
        rejected = rejected[rejected].index

        if verbose: print(f"Rejection {len(rejected)} points")
        if showplot: self._plot(series=s, ok=ok, rejected=rejected, upper_limit=upper_limit, lower_limit=lower_limit)
        return ok, rejected

    def _plot(self, series: Series, ok: DatetimeIndex, rejected: Series, upper_limit: Series, lower_limit: Series):
        # Plot
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot_date(series[ok].index, series[ok], label="OK", color="#4CAF50")
        ax.plot_date(series[rejected].index, series[rejected], marker="X", label="rejected", color="#F44336")
        ax.plot_date(upper_limit.index, upper_limit, marker="x", label="upper limit", color="#F44336")
        ax.plot_date(lower_limit.index, lower_limit, marker="x", label="lower limit", color="#F44336")
        ax.legend()
        fig.show()
