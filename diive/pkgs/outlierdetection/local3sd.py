import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas import DatetimeIndex, Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.repeater import repeater


@ConsoleOutputDecorator()
@repeater
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

    """
    flagid = 'OUTLIER_LOCALSD'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 n_sd: float = 7,
                 winsize: int = None,
                 showplot: bool = False,
                 verbose: bool = False,
                 repeat: bool = True):
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.n_sd = n_sd
        self.winsize = winsize
        self.showplot = showplot
        self.verbose = verbose
        self.repeat = repeat

    def calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Working data
        s = self.series.copy()
        s = s.dropna()

        if not self.winsize:
            winsize = int(len(s) / 20)

        rmedian = s.rolling(window=self.winsize, center=True, min_periods=3).median()
        rsd = s.rolling(window=self.winsize, center=True, min_periods=3).std()
        upper_limit = rmedian + (rsd * self.n_sd)
        lower_limit = rmedian - (rsd * self.n_sd)

        ok = (s < upper_limit) & (s > lower_limit)
        ok = ok[ok].index
        rejected = (s > upper_limit) | (s < lower_limit)
        rejected = rejected[rejected].index

        if self.verbose:
            if self.verbose:
                print(f"Total found outliers: {len(rejected)} values")
        if self.showplot:
            plottitle = f"Outlier detection based on the standard deviation in a rolling window for {self.series.name}"
            self._plot(s, rmedian, upper_limit, lower_limit, plottitle)
            self.plot(ok, rejected, plottitle=plottitle)
        return ok, rejected

    def _plot(self, series, rmedian, upper_limit, lower_limit, plottitle):
        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot_date(series.index, series, label=f"{self.series.name}", color="#42A5F5",
                     alpha=.5, markersize=2, markeredgecolor='none')
        ax.plot_date(rmedian.index, rmedian, label=f"rolling median", color="#FFA726",
                     alpha=.5, markersize=2, markeredgecolor='none')
        ax.plot_date(upper_limit.index, upper_limit, label=f"upper limit", color="#EF5350",
                     alpha=.5, markersize=2, markeredgecolor='none')
        ax.plot_date(lower_limit.index, lower_limit, label=f"lower limit", color="#AB47BC",
                     alpha=.5, markersize=2, markeredgecolor='none')
        default_format(ax=ax)
        default_legend(ax=ax, ncol=2, markerscale=5)
        fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.show()
