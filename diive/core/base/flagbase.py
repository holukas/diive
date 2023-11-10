"""

BASE CLASS FOR QUALITY FLAGS

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

import diive.core.plotting.styles.LightTheme as theme
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.plotfuncs import default_format, default_legend


class FlagBase:

    def __init__(self, series: Series, flagid: str, idstr: str = None, verbose: int = 1):
        self.series = series
        self._flagid = flagid
        self._idstr = validate_id_string(idstr=idstr)
        self.verbose = verbose

        self.flagname = self._generate_flagname()

        self._filteredseries = None
        self._flag = None

        print(f"Generating flag {self.flagname} for variable {self.series.name} ...")

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
        if isinstance(rejected, DatetimeIndex):
            self._flag.loc[rejected] = 2
        else:
            # If no rejected values were found, pass because otherwise
            # NaT is inserted I think at the end of the flag series.
            pass

    def setfiltered(self, rejected: DatetimeIndex):
        """Set rejected values to missing"""
        # Only when rejected exists. If there are no rejected values
        # and the lines here would be executed, it would attach an
        # additional line to the series with NaT as index and Nan
        # as value.
        if isinstance(rejected, DatetimeIndex):
            self._filteredseries.loc[rejected] = np.nan

    def reset(self):
        self._filteredseries = self.series.copy()
        # Generate flag series with NaNs
        self._flag = pd.Series(index=self.series.index, data=np.nan, name=self.flagname)

    def _generate_flagname(self) -> str:
        """Generate standardized name for flag variable"""
        flagname = "FLAG"
        if self._idstr:
            flagname += f"{self._idstr}"
        flagname += f"_{self.series.name}"
        if self._flagid:
            flagname += f"_{self._flagid}"
        flagname += f"_TEST"
        return flagname

    def plot(self, ok: DatetimeIndex, rejected: DatetimeIndex, plottitle: str = ""):
        """Basic plot that shows time series with and without outliers"""
        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax_series = fig.add_subplot(gs[0, 0])
        ax_ok = fig.add_subplot(gs[1, 0], sharex=ax_series)
        ax_series.plot_date(self.series.index, self.series, label=f"{self.series.name}", color="#42A5F5",
                            alpha=.5, markersize=2, markeredgecolor='none')
        ax_series.plot_date(self.series[rejected].index, self.series[rejected],
                            label="outlier (rejected)", color="#F44336", marker="X", alpha=1,
                            markersize=8, markeredgecolor='none')
        ax_ok.plot_date(self.series[ok].index, self.series[ok], label=f"OK", color="#9CCC65", alpha=.5,
                        markersize=2, markeredgecolor='none')
        default_format(ax=ax_series)
        default_format(ax=ax_ok)
        default_legend(ax=ax_series)
        default_legend(ax=ax_ok)
        plt.setp(ax_series.get_xticklabels(), visible=False)
        fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.show()
