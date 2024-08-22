"""

BASE CLASS FOR QUALITY FLAGS

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex
from pandas import Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.histogram import HistogramPlot
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks


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
        gs = gridspec.GridSpec(2, 2)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax_series = fig.add_subplot(gs[0, 0])
        ax_series_hist = fig.add_subplot(gs[0, 1])
        ax_ok = fig.add_subplot(gs[1, 0], sharex=ax_series)
        ax_ok_hist = fig.add_subplot(gs[1, 1])

        ax_series.plot_date(self.series.index, self.series,
                            label=f"{self.series.name}", color="#607D8B",
                            alpha=.5, markersize=8, markeredgecolor='none')
        ax_series.plot_date(self.series[rejected].index, self.series[rejected],
                            label="outlier (rejected)", color="#F44336", alpha=1,
                            markersize=12, markeredgecolor='none', fmt='X')
        hist_kwargs = dict(method='n_bins', n_bins=None, highlight_peak=True, show_zscores=True, show_info=False,
                           show_title=False, show_zscore_values=False, show_grid=False)
        HistogramPlot(self.series, **hist_kwargs).plot(ax=ax_series_hist)

        ax_ok.plot_date(self.series[ok].index, self.series[ok],
                        label="filtered series", alpha=.5,
                        markersize=8, markeredgecolor='none')
        HistogramPlot(self.series[ok], **hist_kwargs).plot(ax=ax_ok_hist)

        default_format(ax=ax_series)
        default_format(ax=ax_ok)
        default_legend(ax=ax_series)
        default_legend(ax=ax_ok)
        plt.setp(ax_series.get_xticklabels(), visible=False)
        plottitle = (f"{self.series.name} filtered by {self.overall_flag.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {n_outliers}")
        nice_date_ticks(ax=ax_series)
        nice_date_ticks(ax=ax_ok)
        fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.tight_layout()
        fig.show()

    def plot_outlier_daytime_nighttime(self, series: Series, flag_daytime: Series,
                                       flag_quality: Series, title: str = None):
        """Plot outlier and non-outlier time series for daytime and nighttime data."""
        # Collect in dataframe for outlier daytime/nighttime plot
        frame = {
            'UNFILTERED': series,
            'UNFILTERED_DT': series[flag_daytime == 1],
            'UNFILTERED_NT': series[flag_daytime == 0],
            'CLEANED': series[flag_quality == 0],
            'CLEANED_DT': series[(flag_quality == 0) & (flag_daytime == 1)],
            'CLEANED_NT': series[(flag_quality == 0) & (flag_daytime == 0)],
            'OUTLIER': series[flag_quality == 2],
            'OUTLIER_DT': series[(flag_quality == 2) & (flag_daytime == 1)],
            'OUTLIER_NT': series[(flag_quality == 2) & (flag_daytime == 0)],
        }
        df = pd.DataFrame(frame)

        fig = plt.figure(facecolor='white', figsize=(24, 12))
        gs = gridspec.GridSpec(3, 4)  # rows, cols
        # gs.update(wspace=0.15, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
        # gs.update(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.5)

        if title:
            fig.suptitle(title, fontsize=24, fontweight='bold')

        ax_series = fig.add_subplot(gs[0, 0])
        ax_series_hist = fig.add_subplot(gs[0, 1])
        ax_cleaned = fig.add_subplot(gs[0, 2], sharex=ax_series)
        ax_cleaned_hist = fig.add_subplot(gs[0, 3])

        ax_series_dt = fig.add_subplot(gs[1, 0])
        ax_series_dt_hist = fig.add_subplot(gs[1, 1])
        ax_cleaned_dt = fig.add_subplot(gs[1, 2], sharex=ax_series)
        ax_cleaned_dt_hist = fig.add_subplot(gs[1, 3])

        ax_series_nt = fig.add_subplot(gs[2, 0], sharex=ax_series)
        ax_series_nt_hist = fig.add_subplot(gs[2, 1])
        ax_cleaned_nt = fig.add_subplot(gs[2, 2], sharex=ax_series)
        ax_cleaned_nt_hist = fig.add_subplot(gs[2, 3])

        axes_series = [ax_series, ax_cleaned, ax_series_dt, ax_cleaned_dt, ax_series_nt, ax_cleaned_nt]
        axes_hist = [ax_series_hist, ax_cleaned_hist, ax_series_dt_hist,
                     ax_cleaned_dt_hist, ax_series_nt_hist, ax_cleaned_nt_hist]
        hist_kwargs = dict(method='n_bins', n_bins=None, highlight_peak=True, show_zscores=True, show_info=False,
                           show_title=False, show_zscore_values=False, show_grid=False)
        series_kwargs = dict(x=df.index, fmt='o', mec='none', alpha=.2, color='black')

        # Column 0
        ax_series.plot_date(
            y=df['CLEANED'], label=f"OK ({df['CLEANED'].count()} values)", **series_kwargs)
        ax_series.plot_date(
            x=df.index, y=df['OUTLIER'], fmt='X', ms=10, mec='none',
            alpha=.9, color='red', label=f"outlier ({df['OUTLIER'].count()} values)")
        ax_series_dt.plot_date(
            y=df['UNFILTERED_DT'], label=f"series ({df['UNFILTERED_DT'].count()} values)", **series_kwargs)
        ax_series_dt.plot_date(
            x=df.index, y=df['OUTLIER_DT'], fmt='X', ms=10, mec='none',
            alpha=.9, color='red', label=f"outlier ({df['OUTLIER_DT'].count()} values)")
        ax_series_nt.plot_date(
            y=df['UNFILTERED_NT'], label=f"series ({df['UNFILTERED_NT'].count()} values)", **series_kwargs)
        ax_series_nt.plot_date(
            x=df.index, y=df['OUTLIER_NT'], fmt='X', ms=10, mec='none',
            alpha=.9, color='red', label=f"outlier ({df['OUTLIER_NT'].count()} values)")

        # Column 1
        HistogramPlot(s=df['UNFILTERED'], **hist_kwargs).plot(ax=ax_series_hist)
        HistogramPlot(s=df['UNFILTERED_DT'], **hist_kwargs).plot(ax=ax_series_dt_hist)
        HistogramPlot(s=df['UNFILTERED_NT'], **hist_kwargs).plot(ax=ax_series_nt_hist)

        # Column 2
        ax_cleaned.plot_date(
            y=df['CLEANED'], label=f"cleaned ({df['CLEANED'].count()} values)", **series_kwargs)
        ax_cleaned_dt.plot_date(
            y=df['CLEANED_DT'], label=f"cleaned daytime ({df['CLEANED_DT'].count()} values)", **series_kwargs)
        ax_cleaned_nt.plot_date(
            y=df['CLEANED_NT'], label=f"cleaned nighttime ({df['CLEANED_NT'].count()} values)", **series_kwargs)

        # Column 3
        HistogramPlot(s=df['CLEANED'], **hist_kwargs).plot(ax=ax_cleaned_hist)
        HistogramPlot(s=df['CLEANED_DT'], **hist_kwargs).plot(ax=ax_cleaned_dt_hist)
        HistogramPlot(s=df['CLEANED_NT'], **hist_kwargs).plot(ax=ax_cleaned_nt_hist)

        for a in axes_series:
            default_format(ax=a, ax_ylabel_txt="value")
            default_legend(ax=a, ncol=1, loc=2)
            nice_date_ticks(ax=a)

        # plt.setp(ax_series.get_xticklabels(), visible=False)
        # plt.setp(ax_cleaned.get_xticklabels(), visible=False)
        # plt.setp(ax_cleaned_dt.get_xticklabels(), visible=False)
        # plt.setp(ax_cleaned_nt.get_xticklabels(), visible=False)
        # plt.setp(ax_daytime.get_xticklabels(), visible=False)

        fig.tight_layout()
        fig.show()
