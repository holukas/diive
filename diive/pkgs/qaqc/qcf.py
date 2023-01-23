"""

==================================
QCF - OVERALL QUALITY CONTROL FLAG
==================================

Combine multiple flags in one single QCF flag.

"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.plotting.heatmap_datetime import HeatmapDateTime
import diive.core.plotting.plotfuncs as pf

class MeteoQCF:
    """Calculate overall quality flag QCF"""

    def __init__(self,
                 flags_df: DataFrame,
                 series: Series,
                 missingflag: str
                 ):
        self.df = flags_df.copy()
        self.series = series
        self.missingflag = missingflag

        self.seriesqcfcol = f"{self.series.name}_QCF"  # Quality-controlled flux
        self.flagqcfcol = f"FLAG_{self.series.name}_QCF"  # Overall flag
        self.sumflags_col = f'SUM_{self.series.name}_FLAGS'
        self.sumhardflags_col = f'SUM_{self.series.name}_HARDFLAGS'
        self.sumsoftflags_col = f'SUM_{self.series.name}_SOFTFLAGS'

        self._flags_df = flags_df.copy()

    @property
    def flags(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._flags_df, DataFrame):
            raise Exception('Results for flags are empty')
        return self._flags_df

    @property
    def seriesqcf(self) -> Series:
        """Return series with rejected values set to missing"""
        return self.flags[self.seriesqcfcol]

    @property
    def flagqcf(self) -> Series:
        """Return QCF flag for series"""
        return self.flags[self.flagqcfcol]

    def calculate(self):
        self._flags_df = self._calculate_flagsums(df=self._flags_df)
        self._flags_df = self._calculate_flag_qcf(df=self._flags_df)
        self._add_series()
        self._calculate_series_qcf()

    def _add_series(self):
        self._flags_df[self.series.name] = self.series.copy()

    def _calculate_series_qcf(self):
        """Create quality-checked time series"""
        self._flags_df[self.seriesqcfcol] = self._flags_df[self.series.name].copy()
        self._flags_df[self.seriesqcfcol].loc[self._flags_df[self.flagqcfcol] == 2] = np.nan

    def _identify_flagcols(self) -> list:
        return [t for t in self.flags.columns
                if str(t).startswith('FLAG_')
                and (str(t).endswith(('_TEST', '_QCF')))
                and str(self.series.name) in str(t)]

    def report_flags(self):

        flagcols = self._identify_flagcols()

        # Report for individual flags
        print(f"\n{'=' * 40}\nREPORT: FLAGS INCL. MISSING VALUES\n{'=' * 40}")
        print("Stats with missing values in the dataset")
        for col in flagcols:
            self._flagstats(col=col, df=self.flags)

        # Report for available values (missing values are ignored)
        # Flags
        print(f"\n{'=' * 40}\nREPORT: FLAGS FOR AVAILABLE RECORDS\n{'=' * 40}")
        print("Stats after removal of missing values")
        df = self.flags.copy()
        ix_missing_vals = df[self.missingflag] == 2
        df = df[~ix_missing_vals].copy()
        for col in flagcols:
            self._flagstats(col=col, df=df)

    def report_qcf_evolution(self):
        """Apply multiple test flags sequentially"""
        # QCF flag evolution
        print(f"\n\n{'=' * 40}\nQCF FLAG EVOLUTION\n{'=' * 40}\n"
              f"This output shows the evolution of the QCF overall quality flag\n"
              f"when test flags are applied sequentially to the variable {self.series.name}.")
        df = self.flags.copy()
        ix_missing_vals = df[self.missingflag] == 2
        df = df[~ix_missing_vals].copy()  # Ignore missing values
        flagcols = self._identify_flagcols()
        allflags_df = df[flagcols].copy()
        n_tests = len(flagcols) + 1  # +1 b/c for loop
        ix_first_test = 0
        n_vals_total_rejected = 0
        n_flag2 = 0
        perc_flag2 = 0
        n_vals = len(allflags_df)
        print(f"\nNumber of {self.series.name} records before QC: {n_vals}")
        for ix_last_test in range(1, n_tests):
            prog_testcols = flagcols[ix_first_test:ix_last_test]
            prog_df = allflags_df[prog_testcols].copy()

            # Calculate QCF (so far)
            prog_df = self._calculate_flagsums(df=prog_df)
            prog_df = self._calculate_flag_qcf(df=prog_df)

            n_flag0 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 0].count()
            n_flag1 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 1].count()
            n_flag2 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 2].count()

            n_vals_test_rejected = n_flag2 - n_vals_total_rejected
            perc_vals_test_rejected = (n_vals_test_rejected / n_vals) * 100
            perc_flag0 = (n_flag0 / n_vals) * 100
            perc_flag1 = (n_flag1 / n_vals) * 100
            perc_flag2 = (n_flag2 / n_vals) * 100

            print(f"+++ {prog_testcols[ix_last_test - 1]} rejected {n_vals_test_rejected} values "
                  f"(+{perc_vals_test_rejected:.2f}%)      "
                  f"TOTALS: flag 0: {n_flag0} ({perc_flag0:.2f}%) / "
                  f"flag 1: {n_flag1} ({perc_flag1:.2f}%) / "
                  f"flag 2: {n_flag2} ({perc_flag2:.2f}%)")

            n_vals_total_rejected = n_flag2

        print(f"\nIn total, {n_flag2} ({perc_flag2:.2f}%) of the available records were rejected in this step.")
        # print(f"\n| Note that this is not the final QC step. More values need to be \n"
        #       f"| rejected after storage correction (Level-3.1) during outlier\n"
        #       f"| removal (Level-3.2) and USTAR filtering (Level-3.3).")

    def _flagstats(self, col: str, df: DataFrame):
        print(f"{col}:")
        flag = df[col]
        n_values = len(flag)
        flagcounts = flag.groupby(flag).count()
        flagmissing = flag.isnull().sum()
        flagmissing_perc = (flagmissing / n_values) * 100
        for flagvalue in flagcounts.index:
            _counts = flagcounts[flagvalue]
            _counts_perc = (_counts / n_values) * 100
            print(f"    OVERALL flag {flagvalue}: {_counts} values ({_counts_perc:.2f}%)  ")
        print(f"    OVERALL flag missing: {flagmissing} values ({flagmissing_perc:.2f}%)  ")
        print("")

    def report_series(self):

        print(f"\n\n{'=' * 40}\nSUMMARY: {self.flagqcfcol}, QCF FLAG FOR {self.series.name}\n{'=' * 40}")

        series = self.flags[self.series.name]
        seriesqcf = self.flags[self.seriesqcfcol]

        n_potential = len(series)
        n_measured = len(series.dropna())
        n_missed = n_potential - n_measured
        n_available = len(seriesqcf.dropna())  # Available after QC
        n_rejected = n_measured - n_available  # Rejected measured values

        perc_measured = (n_measured / n_potential) * 100
        perc_missed = (n_missed / n_potential) * 100
        perc_available = (n_available / n_measured) * 100
        perc_rejected = (n_rejected / n_measured) * 100

        start = series.index[0].strftime('%Y-%m-%d %H:%M')
        end = series.index[-1].strftime('%Y-%m-%d %H:%M')
        print(f"Between {start} and {end} ...\n"
              f"    Total flux records BEFORE quality checks: {n_measured} ({perc_measured:.2f}% of potential)\n"
              f"    Available flux records AFTER quality checks: {n_available} ({perc_available:.2f}% of total)\n"
              f"    Rejected flux records: {n_rejected} ({perc_rejected:.2f}% of total)\n"
              f"    Potential flux records: {n_potential}\n"
              f"    Potential flux records missed: {n_missed} ({perc_missed:.2f}% of potential)\n")

    def _calculate_flag_qcf(self, df: DataFrame) -> DataFrame:
        """Calculate overall QCF flag from flag sums"""
        # QCF is NaN if no flag is available
        df[self.flagqcfcol] = np.nan

        # QCF is 0 if all flags show zero
        df[self.flagqcfcol].loc[df[self.sumflags_col] == 0] = 0

        # tests[QCF].loc[tests['SUM_FLAGS'] == 1] = 1

        # QCF is 2 if three soft flags were raised
        df[self.flagqcfcol].loc[df[self.sumsoftflags_col] > 3] = 2

        # QCF is 2 if at least one hard flag was raised
        df[self.flagqcfcol].loc[df[self.sumhardflags_col] >= 2] = 2

        # QCF is 1 if no hard flag and max. three soft flags and
        # min. one soft flag were raised
        df[self.flagqcfcol].loc[(df[self.sumsoftflags_col] <= 3)
                                & (df[self.sumsoftflags_col] >= 1)
                                & (df[self.sumhardflags_col] == 0)] = 1

        return df

    def _calculate_flagsums(self, df: DataFrame) -> DataFrame:
        """Calculate sums of all individual flags"""
        sumhardflags = df[df == 2].sum(axis=1)  # The sum of all flags that show 2
        sumsoftflags = df[df == 1].sum(axis=1)  # The sum of all flags that show 1
        sumflags = sumhardflags.add(sumsoftflags)  # Sum of all flags
        df[self.sumhardflags_col] = sumhardflags
        df[self.sumsoftflags_col] = sumsoftflags
        df[self.sumflags_col] = sumflags
        return df

    def showplot_heatmaps(self, maxflux: float = None):

        fig = plt.figure(facecolor='white', figsize=(20, 15))
        gs = gridspec.GridSpec(1, 4)  # rows, cols
        gs.update(wspace=0.3, hspace=0.5, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1], sharey=ax_before)
        ax_flagsum = fig.add_subplot(gs[0, 2], sharey=ax_before)
        ax_flag = fig.add_subplot(gs[0, 3], sharey=ax_before)

        # Heatmaps
        vmin = -maxflux if maxflux else None
        vmax = maxflux if maxflux else None
        HeatmapDateTime(ax=ax_before, series=self.flags[self.series.name], vmin=vmin, vmax=vmax,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_after, series=self.flags[self.seriesqcfcol], vmin=vmin, vmax=vmax,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flagsum, series=self.flags[self.sumflags_col],
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flag, series=self.flags[self.flagqcfcol],
                        cb_digits_after_comma=0).plot()
        plt.setp(ax_after.get_yticklabels(), visible=False)
        plt.setp(ax_flagsum.get_yticklabels(), visible=False)
        plt.setp(ax_flag.get_yticklabels(), visible=False)
        ax_after.axes.get_yaxis().get_label().set_visible(False)
        ax_flagsum.axes.get_yaxis().get_label().set_visible(False)
        ax_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()

    def showplot_timeseries(self):
        self.flags.plot(subplots=True, figsize=(20,15))
        plt.show()


class FluxQCF:

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 level: float,
                 swinpotcol: str,
                 nighttime_threshold: int = 50,
                 daytime_accept_qcf_below: int = 2,
                 nighttimetime_accept_qcf_below: int = 1):
        self.df = df.copy()
        self.fluxcol = fluxcol
        self.level = level
        self.swinpotcol = swinpotcol
        self.nighttime_threshold = nighttime_threshold
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below

        self.fluxqcfcol = f"{self.fluxcol}_QCF"  # Quality-controlled flux
        self.flagqcfcol = f"FLAG_{self.fluxcol}_QCF"  # Overall flag
        self.sumflags_col = f'SUM_{self.fluxcol}_FLAGS'
        self.sumhardflags_col = f'SUM_{self.fluxcol}_HARDFLAGS'
        self.sumsoftflags_col = f'SUM_{self.fluxcol}_SOFTFLAGS'
        self.daytimecol = 'DAYTIME'
        self.nighttimecol = 'NIGHTTIME'
        self.level2missingcol = self._detect_name_missing_test()

        self._flags_df = self._initial_subset(df=self.df)

    @property
    def flags_df(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._flags_df, DataFrame):
            raise Exception('Results for flux flags are empty')
        return self._flags_df

    def get(self) -> DataFrame:
        """Return original data with QCF flag"""
        _df = self.df.copy()  # Main data
        newcols = [col for col in self.flags_df.columns if col not in self.df]
        newcols_df = self._flags_df[newcols].copy()
        _df = pd.concat([_df, newcols_df], axis=1)  # Add new columns to main data
        [print(f"++Adding new column {c} (Level-{self.level} overall quality flag) to main data ...") for c in newcols]
        return _df

    def calculate(self):
        self._flags_df = self._calculate_flagsums(df=self._flags_df)
        self._flags_df = self._calculate_flag_qcf(df=self._flags_df)
        self._calculate_flux_qcf()

    def _calculate_flux_qcf(self):
        """Create quality-checked time series"""
        self._flags_df[self.fluxqcfcol] = self._flags_df[self.fluxcol].copy()
        self._flags_df[self.fluxqcfcol].loc[self._flags_df[self.flagqcfcol] == 2] = np.nan

    def report_flags(self):

        test_cols = [t for t in self.flags_df.columns if 'FLAG' in str(t)]

        # Report for individual flags
        print(f"\n{'=' * 40}\nREPORT: FLUX FLAGS INCL. MISSING VALUES\n{'=' * 40}")
        print("Stats with missing values in the dataset")
        for col in test_cols:
            self._flagstats_dt_nt(col=col, df=self.flags_df)

        # Report for available values (missing values are ignored)
        # Flags
        print(f"\n{'=' * 40}\nREPORT: FLUX FLAGS FOR AVAILABLE RECORDS\n{'=' * 40}")
        print("Stats after removal of missing values")
        df = self.flags_df.copy()
        ix_missing_vals = df[self.level2missingcol] == 2
        df = df[~ix_missing_vals].copy()
        for col in test_cols:
            self._flagstats_dt_nt(col=col, df=df)

    def report_qcf_evolution(self):
        """Apply multiple test flags sequentially"""
        # QCF flag evolution
        print(f"\n\n{'=' * 40}\nQCF FLAG EVOLUTION\n{'=' * 40}\n"
              f"Swiss FluxNet processing chain, Level-2: Quality flag expansion\n"
              f"This output shows the evolution of the QCF overall quality flag\n"
              f"when test flags from the EddyPro output are applied sequentially\n"
              f"to the flux variable {self.fluxcol}.")
        df = self.flags_df.copy()
        ix_missing_vals = df[self.level2missingcol] == 2
        df = df[~ix_missing_vals].copy()  # Ignore missing values
        flagcols = [c for c in df.columns if str(c).startswith('FLAG_L')]
        allflags_df = df[flagcols].copy()
        n_tests = len(flagcols) + 1  # +1 b/c for loop
        ix_first_test = 0
        n_vals_total_rejected = 0
        n_flag2 = 0
        perc_flag2 = 0
        n_vals = len(allflags_df)
        print(f"\nNumber of {self.fluxcol} records before QC: {n_vals}")
        for ix_last_test in range(1, n_tests):
            prog_testcols = flagcols[ix_first_test:ix_last_test]
            prog_df = allflags_df[prog_testcols].copy()

            # Calculate QCF (so far)
            prog_df = self._calculate_flagsums(df=prog_df)
            prog_df[self.daytimecol] = df[self.daytimecol].copy()  # QCF calc needs radiation
            prog_df[self.nighttimecol] = df[self.nighttimecol].copy()  # QCF calc needs radiation
            prog_df = self._calculate_flag_qcf(df=prog_df)

            n_flag0 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 0].count()
            n_flag1 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 1].count()
            n_flag2 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 2].count()

            n_vals_test_rejected = n_flag2 - n_vals_total_rejected
            perc_vals_test_rejected = (n_vals_test_rejected / n_vals) * 100
            perc_flag0 = (n_flag0 / n_vals) * 100
            perc_flag1 = (n_flag1 / n_vals) * 100
            perc_flag2 = (n_flag2 / n_vals) * 100

            print(f"+++ {prog_testcols[ix_last_test - 1]} rejected {n_vals_test_rejected} values "
                  f"(+{perc_vals_test_rejected:.2f}%)      "
                  f"TOTALS: flag 0: {n_flag0} ({perc_flag0:.2f}%) / "
                  f"flag 1: {n_flag1} ({perc_flag1:.2f}%) / "
                  f"flag 2: {n_flag2} ({perc_flag2:.2f}%)")

            n_vals_total_rejected = n_flag2

        print(f"\nIn total, {n_flag2} ({perc_flag2:.2f}%) of the available records were rejected in this step.")
        # print(f"\n| Note that this is not the final QC step. More values need to be \n"
        #       f"| rejected after storage correction (Level-3.1) during outlier\n"
        #       f"| removal (Level-3.2) and USTAR filtering (Level-3.3).")

    def _flagstats_dt_nt(self, col: str, df: DataFrame):
        if (str(col) != self.fluxcol) & (str(col) != self.swinpotcol):
            print(f"{col}:")
            flag = df[col]
            self._flagstats(flag=flag, prefix="OVERALL")
            flag = df[col].loc[df[self.daytimecol] == 1]
            self._flagstats(flag=flag, prefix="DAYTIME")
            flag = df[col].loc[df[self.nighttimecol] == 1]
            self._flagstats(flag=flag, prefix="NIGHTTIME")

    def _flagstats(self, flag: Series, prefix: str):
        n_values = len(flag)
        flagcounts = flag.groupby(flag).count()
        flagmissing = flag.isnull().sum()
        flagmissing_perc = (flagmissing / n_values) * 100
        for flagvalue in flagcounts.index:
            _counts = flagcounts[flagvalue]
            _counts_perc = (_counts / n_values) * 100
            print(f"    {prefix} flag {flagvalue}: {_counts} values ({_counts_perc:.2f}%)  ")
        print(f"    {prefix} flag missing: {flagmissing} values ({flagmissing_perc:.2f}%)  ")
        print("")

    def _add_daytime_info(self, df: DataFrame) -> DataFrame:
        """Separate columns to indicate daytime or nighttime"""
        df[self.daytimecol] = np.nan
        df[self.daytimecol].loc[df[self.swinpotcol] >= self.nighttime_threshold] = 1
        df[self.daytimecol].loc[df[self.swinpotcol] < self.nighttime_threshold] = 0
        df[self.nighttimecol] = np.nan
        df[self.nighttimecol].loc[df[self.swinpotcol] >= self.nighttime_threshold] = 0
        df[self.nighttimecol].loc[df[self.swinpotcol] < self.nighttime_threshold] = 1
        return df

    def report_flux(self):

        print(f"\n\n{'=' * 40}\nSUMMARY: {self.flagqcfcol}, QCF FLAG FOR {self.fluxcol}\n{'=' * 40}")

        flux = self.flags_df[self.fluxcol]
        fluxqcf = self.flags_df[self.fluxqcfcol]

        n_potential = len(flux)
        n_measured = len(flux.dropna())
        n_missed = n_potential - n_measured
        n_available = len(fluxqcf.dropna())  # Available after QC
        n_rejected = n_measured - n_available  # Rejected measured values

        perc_measured = (n_measured / n_potential) * 100
        perc_missed = (n_missed / n_potential) * 100
        perc_available = (n_available / n_measured) * 100
        perc_rejected = (n_rejected / n_measured) * 100

        start = flux.index[0].strftime('%Y-%m-%d %H:%M')
        end = flux.index[1].strftime('%Y-%m-%d %H:%M')
        print(f"Between {start} and {end} ...\n"
              f"    Total flux records BEFORE quality checks: {n_measured} ({perc_measured:.2f}% of potential)\n"
              f"    Available flux records AFTER quality checks: {n_available} ({perc_available:.2f}% of total)\n"
              f"    Rejected flux records: {n_rejected} ({perc_rejected:.2f}% of total)\n"
              f"    Potential flux records: {n_potential}\n"
              f"    Potential flux records missed: {n_missed} ({perc_missed:.2f}% of potential)\n")

    def _detect_flagcols(self, df: DataFrame) -> list:
        flagcols = [c for c in df.columns if str(c).startswith('FLAG_L')]
        relevant = None
        fluxcol = str(self.fluxcol)
        if fluxcol.startswith('NEE_'):
            relevant = ['_FC_', '_NEE_']
        elif fluxcol == 'FC':
            relevant = ['_FC_']
        elif fluxcol.startswith('H_') or fluxcol == 'H':
            relevant = ['_H_']
        elif fluxcol.startswith('LE_') or fluxcol == 'LE':
            relevant = ['_LE_']
        elif fluxcol.startswith('ET_') or fluxcol == 'ET':
            relevant = ['_ET_']
        elif fluxcol.startswith('FH2O_') or fluxcol == 'FH2O':
            relevant = ['_FH2O_']
        elif fluxcol.startswith('TAU_') or fluxcol == 'TAU':
            relevant = ['_TAU_']
        flagcols = [f for f in flagcols if any(n in f for n in relevant)]
        # flagcols = [c for c in flagcols if f'_{self.fluxcol}_' in c]  # Old method
        return flagcols

    def _detect_name_missing_test(self):
        """Detect relevant Level-2 missing test for flux"""
        relevant = None
        fluxcol = str(self.fluxcol)
        if fluxcol.startswith('NEE_') or fluxcol == 'FC':
            relevant = 'FC'
        elif fluxcol.startswith('H_') or fluxcol == 'H':
            relevant = 'H'
        elif fluxcol.startswith('LE_') or fluxcol == 'LE':
            relevant = 'LE'
        elif fluxcol.startswith('ET_') or fluxcol == 'ET':
            relevant = 'ET'
        elif fluxcol.startswith('FH2O_') or fluxcol == 'FH2O':
            relevant = 'H2O'
        elif fluxcol.startswith('TAU_') or fluxcol == 'TAU':
            relevant = 'TAU'
        level2missingcol = f'FLAG_L2_{relevant}_MISSING_TEST'
        return level2missingcol

    def _calculate_flag_qcf(self, df: DataFrame) -> DataFrame:
        """Calculate overall QCF flag from flag sums"""
        # QCF is NaN if no flag is available
        df[self.flagqcfcol] = np.nan

        # QCF is 0 if all flags show zero
        df[self.flagqcfcol].loc[df[self.sumflags_col] == 0] = 0

        # tests[QCF].loc[tests['SUM_FLAGS'] == 1] = 1

        # QCF is 2 if three soft flags were raised
        df[self.flagqcfcol].loc[df[self.sumsoftflags_col] > 3] = 2

        # QCF is 2 if at least one hard flag was raised
        df[self.flagqcfcol].loc[df[self.sumhardflags_col] >= 2] = 2

        # QCF is 1 if no hard flag and max. three soft flags and
        # min. one soft flag were raised
        df[self.flagqcfcol].loc[(df[self.sumsoftflags_col] <= 3)
                                & (df[self.sumsoftflags_col] >= 1)
                                & (df[self.sumhardflags_col] == 0)] = 1

        # Remove daytime values based on param
        df[self.flagqcfcol].loc[(df[self.flagqcfcol] >= self.daytime_accept_qcf_below)
                                & (df[self.daytimecol] == 1)] = 2

        # Remove nighttime values based on param
        df[self.flagqcfcol].loc[(df[self.flagqcfcol] >= self.nighttimetime_accept_qcf_below)
                                & (df[self.nighttimecol] == 1)] = 2
        return df

    def _calculate_flagsums(self, df: DataFrame) -> DataFrame:
        """Calculate sums of all individual flags"""
        flagcols = self._detect_flagcols(df=df)
        subset = df[flagcols].copy()
        sumhardflags = subset[subset == 2].sum(axis=1)  # The sum of all flags that show 2
        sumsoftflags = subset[subset == 1].sum(axis=1)  # The sum of all flags that show 1
        sumflags = sumhardflags.add(sumsoftflags)  # Sum of all flags
        df[self.sumhardflags_col] = sumhardflags
        df[self.sumsoftflags_col] = sumsoftflags
        df[self.sumflags_col] = sumflags
        return df

    def showplot(self, maxflux: float):

        fig = plt.figure(facecolor='white', figsize=(19, 9))
        gs = gridspec.GridSpec(1, 4)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1], sharey=ax_before)
        ax_flagsum = fig.add_subplot(gs[0, 2], sharey=ax_before)
        ax_flag = fig.add_subplot(gs[0, 3], sharey=ax_before)

        HeatmapDateTime(ax=ax_before, series=self.flags_df[self.fluxcol], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_after, series=self.flags_df[self.fluxqcfcol], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flagsum, series=self.flags_df[self.sumflags_col],
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flag, series=self.flags_df[self.flagqcfcol],
                        cb_digits_after_comma=0).plot()

        plt.setp(ax_after.get_yticklabels(), visible=False)
        plt.setp(ax_flagsum.get_yticklabels(), visible=False)
        plt.setp(ax_flag.get_yticklabels(), visible=False)

        ax_after.axes.get_yaxis().get_label().set_visible(False)
        ax_flagsum.axes.get_yaxis().get_label().set_visible(False)
        ax_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()

    def _initial_subset(self, df: DataFrame) -> DataFrame:
        """Make subset with flux, SW_IN_POT and flags"""
        subset = df[[self.fluxcol, self.swinpotcol]].copy()
        subset = self._add_daytime_info(df=subset)
        flagcols = self._detect_flagcols(df=df)
        subset = pd.concat([subset, df[flagcols]], axis=1)
        return subset


def findqcfcol(df: DataFrame, varname: str) -> str:
    """
    Find name of column that contains the QCF for the
    selected variable

    (QCF=quality control flag)

    Args:
        df: DataFrame that contains the QCF
        varname: Name of the variable for which the QCF is searched.
            Only used to construct the QCF naming

    Returns:
        Name of the QCF flag found in *df*
    """
    flagqcfcol = [f for f in df.columns
                  if str(f).startswith('FLAG_')
                  and str(f).endswith('_QCF')
                  and varname in f]
    flagqcfcol = flagqcfcol[0] if len(flagqcfcol) == 1 else None
    if not flagqcfcol: raise (f"There is more than one or no QCF flag available for {varname}.")
    return flagqcfcol
