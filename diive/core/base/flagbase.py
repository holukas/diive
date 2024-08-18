"""

BASE CLASS FOR QUALITY FLAGS

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex

import diive.core.plotting.styles.LightTheme as theme
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.plotfuncs import default_format, default_legend


class FlagBase:

    def __init__(self, series: Series, flagid: str, idstr: str = None, verbose: bool = True):
        self.series = series
        self._flagid = flagid
        self._idstr = validate_id_string(idstr=idstr)
        self.verbose = verbose

        self._overall_flag = None
        self._filteredseries = None
        self._flag = None

    @property
    def overall_flag(self) -> Series:
        """Overall flag, calculated from individual flags from multiple iterations."""
        if not isinstance(self._overall_flag, Series):
            raise Exception('No overall flag available.')
        return self._overall_flag

    def get_flag(self):
        return self.overall_flag

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

    def collect_results(self) -> DataFrame:
        """Store flag of this iteration in dataframe."""
        frame = {
            # self.filteredseries.name: self.filteredseries.copy(),
            self.flag.name: self.flag.copy()
        }
        iteration_df = pd.DataFrame.from_dict(frame)
        return iteration_df

    def get_filteredseries(self, iteration) -> Series:
        """For the first iteration the original input series is used,
        for all other iterations the filtered series from the previous
        iteration is used."""
        filteredseries = self.series.copy() if iteration == 1 else self.filteredseries
        # Rename filtered series to include iteration number
        filteredname = self.generate_iteration_filtered_variable_name(iteration=iteration)
        filteredseries.name = filteredname
        return filteredseries

    def init_flag(self, iteration):
        """Initiate (empty) flag for this iteration."""
        flagname = self.generate_flagname(iteration=iteration)
        flag = pd.Series(index=self.filteredseries.index, data=np.nan, name=flagname)
        return flag

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

    def generate_flagname(self, iteration: int = None) -> str:
        """Generate standardized name for flag variable"""
        flagname = "FLAG"
        if self._idstr:
            flagname += f"{self._idstr}"
        flagname += f"_{self.series.name}"
        if self._flagid:
            flagname += f"_{self._flagid}"
        if iteration:
            flagname += f'_ITER{iteration}_TEST'
        else:
            flagname += '_TEST'
        return flagname

    def generate_iteration_filtered_variable_name(self, iteration: int):
        filteredname = f"{self.series.name}_FILTERED-AFTER-ITER{iteration}"
        return filteredname

    def repeat(self, func, repeat):
        """Repeat function until no more outliers found."""
        n_outliers = 9999
        iteration = 0
        iteration_flags_df = pd.DataFrame()
        while n_outliers > 0:
            iteration += 1
            cur_iteration_flag_df, n_outliers = func(iteration=iteration)
            iteration_flags_df = pd.concat([iteration_flags_df, cur_iteration_flag_df], axis=1)
            if not repeat:
                break

        # Calcualte the sum of all flags that show 2, for each data row
        overall_flag = iteration_flags_df[iteration_flags_df == 2].sum(axis=1)
        overall_flag.name = self.generate_flagname()

        n_iterations = len(iteration_flags_df.columns)

        return overall_flag, n_iterations

    def run_flagtests(self, iteration):
        """Calculate flag for given iteration."""
        self._filteredseries = self.get_filteredseries(iteration=iteration)
        self._flag = self.init_flag(iteration=iteration)
        ok, rejected, n_outliers = self._flagtests(iteration=iteration)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)
        iteration_df = self.collect_results()
        return iteration_df, n_outliers

    def defaultplot(self, n_iterations: int = 1):
        """Basic plot that shows time series with and without outliers"""
        ok = self.overall_flag == 0
        rejected = self.overall_flag == 2
        n_outliers = rejected.sum()

        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax_series = fig.add_subplot(gs[0, 0])
        ax_ok = fig.add_subplot(gs[1, 0], sharex=ax_series)
        ax_series.plot_date(self.series.index, self.series,
                            label=f"{self.series.name}", color="#607D8B",
                            alpha=.5, markersize=8, markeredgecolor='none')
        ax_series.plot_date(self.series[rejected].index, self.series[rejected],
                            label="outlier (rejected)", color="#F44336", alpha=1,
                            markersize=12, markeredgecolor='none', fmt='X')
        ax_ok.plot_date(self.series[ok].index, self.series[ok],
                        label="filtered series", alpha=.5,
                        markersize=8, markeredgecolor='none')
        default_format(ax=ax_series)
        default_format(ax=ax_ok)
        default_legend(ax=ax_series)
        default_legend(ax=ax_ok)
        plt.setp(ax_series.get_xticklabels(), visible=False)
        plottitle = (f"{self.series.name} filtered by {self.overall_flag.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {n_outliers}")
        fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.show()


