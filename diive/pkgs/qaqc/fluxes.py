# TODO currently indev
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.plotting.heatmap_datetime import HeatmapDateTime


class FluxQCF:

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 daytime_accept_qcf_below: int = 2,
                 nighttimetime_accept_qcf_below: int = 1):
        """
        Create QCF (quality-control flag) for selected flags, calculated
        from EddyPro's _fluxnet_ output files

        Args:
            df: Dataframe containing data from EddyPro's _fluxnet_ file
            fluxcol: Name of the flux variable in *df*
            daytime_accept_qcf_below:
            nighttimetime_accept_qcf_below:
        """
        self.fluxcol = fluxcol
        self.df = df
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below

        self.fqcfcol = f"{self.fluxcol}_QCF"  # Quality-controlled flux
        self.qcfcol = f"FLAG_{self.fluxcol}_QCF"  # Overall flag
        self.swin_pot_col = "SW_IN_POT"  # Potential radiation
        self.daytimecol = "DAYTIME"
        self.nighttimecol = "NIGHTTIME"

        self.gas = self._detect_gas()

        self._fluxflags = self.df[[fluxcol, self.swin_pot_col]].copy()
        self._add_daytime_info()

        self._fluxqcf = None

    def _add_daytime_info(self):
        # Separate columns to indicate daytime of nighttime
        self._fluxflags[self.daytimecol] = np.nan
        self._fluxflags[self.daytimecol].loc[self._fluxflags[self.swin_pot_col] >= 20] = 1
        self._fluxflags[self.daytimecol].loc[self._fluxflags[self.swin_pot_col] < 20] = 0
        self._fluxflags[self.nighttimecol] = np.nan
        self._fluxflags[self.nighttimecol].loc[self._fluxflags[self.swin_pot_col] >= 20] = 0
        self._fluxflags[self.nighttimecol].loc[self._fluxflags[self.swin_pot_col] < 20] = 1

    @property
    def fluxflags(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._fluxflags, DataFrame):
            raise Exception('Results for flux flags are empty')
        return self._fluxflags

    @property
    def fluxqcf(self) -> DataFrame:
        """Return dataframe containing quality-checked flux"""
        if not isinstance(self._fluxqcf, DataFrame):
            raise Exception('Results for quality-controlled flux are empty')
        return self._fluxqcf

    def calculate_qcf(self):
        self._fluxflags = self._calculate_flagsums(df=self.fluxflags)
        self._fluxflags = self._calculate_qcf(df=self.fluxflags)

        # Create quality-checked time series
        self._fluxflags[self.fqcfcol] = self._fluxflags[self.fluxcol].copy()
        self._fluxflags[self.fqcfcol].loc[self._fluxflags[self.qcfcol] == 2] = np.nan

        # Create nice subset
        cols = [self.fluxcol, self.fqcfcol, self.qcfcol, self.swin_pot_col, self.daytimecol, self.nighttimecol]
        self._fluxqcf = self.fluxflags[cols].copy()

    def report_flags(self):

        test_cols = [t for t in self.fluxflags.columns if str(t).startswith('FLAG_')]

        # Report for individual flags
        print(f"\n{'=' * 40}\nREPORT: FLUX FLAGS INCL. MISSING VALUES\n{'=' * 40}")
        print("Stats with missing values in the dataset")
        for col in test_cols:
            self._flagstats_dt_nt(col=col, df=self.fluxflags)

        # Report for available values (missing values are ignored)
        # Only executed if the QCF variable was created
        if self.qcfcol in self.fluxflags.columns:
            # Flags
            print(f"\n{'=' * 40}\nREPORT: FLUX FLAGS FOR AVAILABLE RECORDS\n{'=' * 40}")
            print("Stats after removal of missing values")
            df = self.fluxflags.copy()
            ix_missing_vals = df[f'FLAG_{self.fluxcol}_MISSING_TEST'] == 2
            df = df[~ix_missing_vals].copy()
            for col in test_cols:
                self._flagstats_dt_nt(col=col, df=df)

    def report_qcf_evolution(self):
        """
        Apply multiple test flags sequentially

        Returns:

        """

        # QCF flag evolution
        print(f"\n\n{'=' * 40}\nQCF FLAG EVOLUTION\n{'=' * 40}\n"
              f"Swiss FluxNet processing chain, Level-2: Quality flag expansion\n"
              f"This output shows the evolution of the QCF overall quality flag\n"
              f"when test flags from the EddyPro output are applied sequentially\n"
              f"to the flux variable {self.fluxcol}.")
        df = self.fluxflags.copy()
        ix_missing_vals = df[f'FLAG_{self.fluxcol}_MISSING_TEST'] == 2
        df = df[~ix_missing_vals].copy()  # Ignore missing values
        flagcols = [c for c in df.columns if str(c).startswith('FLAG_')]
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
            prog_df[self.nighttimecol] = df[self.nighttimecol].copy()  # QCF calc needs radiation
            prog_df = self._calculate_qcf(df=prog_df)

            n_flag0 = prog_df[self.qcfcol].loc[prog_df[self.qcfcol] == 0].count()
            n_flag1 = prog_df[self.qcfcol].loc[prog_df[self.qcfcol] == 1].count()
            n_flag2 = prog_df[self.qcfcol].loc[prog_df[self.qcfcol] == 2].count()

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
        print(f"\n| Note that this is not the final QC step. More values need to be \n"
              f"| rejected after storage correction (Level-3.1) during outlier\n"
              f"| removal (Level-3.2) and USTAR filtering (Level-3.3).")

    def report_flux(self):
        self._fluxstats()

    def _calculate_flagsums(self, df: DataFrame) -> DataFrame:
        # Generate subset with all flags (tests)
        flagcols = [t for t in df.columns if str(t).startswith('FLAG_')]
        subset_tests = df[flagcols].copy()

        # The sum of all flags that show 2
        subset_tests['SUM_HARDFLAGS'] = subset_tests[subset_tests == 2].sum(axis=1)

        # The sum of all flags that show 1
        subset_tests['SUM_SOFTFLAGS'] = subset_tests[subset_tests == 1].sum(axis=1)
        subset_tests['SUM_FLAGS'] = subset_tests['SUM_HARDFLAGS'].add(subset_tests['SUM_SOFTFLAGS'])

        df['SUM_HARDFLAGS'] = subset_tests['SUM_HARDFLAGS'].copy()
        df['SUM_SOFTFLAGS'] = subset_tests['SUM_SOFTFLAGS'].copy()
        df['SUM_FLAGS'] = subset_tests['SUM_FLAGS'].copy()

        return df

    def _calculate_qcf(self, df: DataFrame) -> DataFrame:
        """Calculate QCF from flag sums"""
        # QCF is NaN if no flag is available
        df[self.qcfcol] = np.nan

        # QCF is 0 if all flags show zero
        df[self.qcfcol].loc[df['SUM_FLAGS'] == 0] = 0

        # tests[QCF].loc[tests['SUM_FLAGS'] == 1] = 1

        # QCF is 2 if three soft flags were raised
        df[self.qcfcol].loc[df['SUM_SOFTFLAGS'] > 3] = 2

        # QCF is 2 if at least one hard flag was raised
        df[self.qcfcol].loc[df['SUM_HARDFLAGS'] >= 2] = 2

        # QCF is 1 if no hard flag and max. three soft flags and
        # min. one soft flag were raised
        df[self.qcfcol].loc[(df['SUM_SOFTFLAGS'] <= 3)
                            & (df['SUM_SOFTFLAGS'] >= 1)
                            & (df['SUM_HARDFLAGS'] == 0)] = 1

        # Remove nighttime values where QCF = 1 or QCF = 2
        df[self.qcfcol].loc[(df[self.qcfcol] > 0)
                            & (df['NIGHTTIME'] == 1)] = 2

        return df

    def _fluxstats(self):

        print(f"\n\n{'=' * 40}\nSUMMARY: {self.qcfcol}, QCF FLAG FOR {self.fluxcol}\n{'=' * 40}")

        flux = self.fluxqcf[self.fluxcol]
        fqcf = self.fluxqcf[self.fqcfcol]

        n_potential = len(flux)
        n_measured = len(flux.dropna())
        n_missed = n_potential - n_measured
        n_available = len(fqcf.loc[self.fluxqcf[self.qcfcol] < 2])  # Available after QC
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

    def _flagstats_dt_nt(self, col: str, df: DataFrame):
        if (str(col) != self.fluxcol) & (str(col) != self.swin_pot_col):
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

    def raw_data_screening_vm97(self):
        """

        Flag from EddyPro fluxnet files is an integer and looks like this, e.g.: 801000100
        One integer contains *multiple tests* for *one* gas.

        There is one flag for each gas, which is different from the flag output in the
        EddyPro full output file (there, one integer describes *one test* and then contains
        flags for *multiple gases*).

        _HF_ = hard flag (flag 1 = bad values)
        _SF_ = soft flag (flag 1 = ok values)

        """
        vm97 = f"{self.gas}_VM97_TEST"
        flagdf = self.df[[self.fluxcol, vm97]].copy()

        flagdf[vm97] = flagdf[vm97].apply(pd.to_numeric, errors='coerce').astype(float)
        flagdf[vm97] = flagdf[vm97].fillna(899999999)  # 9 = missing flags

        flagcols = {
            # '0': XXX,  # Index 0 is always the number `8`
            '1': f"FLAG_{self.fluxcol}_{self.gas}_VM97_SPIKE_HF_TEST",  # Spike detection, hard flag
            '2': f"FLAG_{self.fluxcol}_{self.gas}_VM97_AMPLITUDE_RESOLUTION_HF_TEST",  # Amplitude resolution, hard flag
            '3': f"FLAG_{self.fluxcol}_{self.gas}_VM97_DROPOUT_TEST",  # Drop-out, hard flag
            '4': f"FLAG_{self.fluxcol}_{self.gas}_VM97_ABSOLUTE_LIMITS_HF_TEST",  # Absolute limits, hard flag
            '5': f"FLAG_{self.fluxcol}_{self.gas}_VM97_SKEWKURT_HF_TEST",  # Skewness/kurtosis, hard flag
            '6': f"FLAG_{self.fluxcol}_{self.gas}_VM97_SKEWKURT_SF_TEST",  # Skewness/kurtosis, soft flag
            '7': f"FLAG_{self.fluxcol}_{self.gas}_VM97_DISCONTINUITIES_HF_TEST",  # Discontinuities, hard flag
            '8': f"FLAG_{self.fluxcol}_{self.gas}_VM97_DISCONTINUITIES_SF_TEST"  # Discontinuities, soft flag
        }

        # Extract 8 individual flags from VM97 multi-flag integer
        for i, c in flagcols.items():
            flagdf[c] = flagdf[vm97].astype(str).str[int(i)].astype(float) \
                .replace(9, np.nan)
            if '_HF_' in c:
                # Hard flag 1 corresponds to bad value
                flagdf[c] = flagdf[c].replace(1, 2)

        # Make new dict that contains flags that we use later
        flagcols_used = {x: flagcols[x] for x in flagcols if x in ('1', '3', '4')}

        # Collect all required flags
        for i, c in flagcols_used.items():
            self._fluxflags[c] = flagdf[c].copy()
            print(f"Fetching {c} for {self.gas} "
                  f"with flag 0 (good values) where test passed, "
                  f"flag 2 (bad values) where test failed (for hard flags) or "
                  f"flag 1 (ok values) where test failed (for soft flags) ...")

    def signal_strength_test(self,
                             signal_strength_col: str,
                             method: str,
                             threshold: int):
        flagname = f'FLAG_{self.fluxcol}_SIGNAL_STRENGTH_TEST'
        flagdf = self.df[[self.fluxcol, signal_strength_col]].copy()
        flagdf[flagname] = np.nan

        if method == 'discard below':
            print(f"Calculating {flagname} from {signal_strength_col} with "
                  f"flag 0 (good values) where {signal_strength_col} >= {threshold}, "
                  f"flag 2 (bad values) where {signal_strength_col} < {threshold} ...")
        elif method == 'discard above':
            print(f"Calculating {flagname} from {signal_strength_col} with "
                  f"flag 0 (good values) where {signal_strength_col} <= {threshold}, "
                  f"flag 2 (bad values) where {signal_strength_col} > {threshold} ...")

        # percentiles_df = percentiles(series=flagdf[signal_strength_col], showplot=False)

        good = None
        bad = None
        if method == 'discard below':
            good = flagdf[signal_strength_col] >= threshold
            bad = flagdf[signal_strength_col] < threshold
        elif method == 'discard above':
            good = flagdf[signal_strength_col] <= threshold
            bad = flagdf[signal_strength_col] > threshold

        flagdf[flagname][good] = 0
        flagdf[flagname][bad] = 2

        self._fluxflags[flagname] = flagdf[flagname].copy()

    def spectral_correction_factor_test(self,
                                        thres_good: int = 2,
                                        thres_ok: int = 4):
        flagname = f'FLAG_{self.fluxcol}_SCF_TEST'
        scf = f'{self.fluxcol}_SCF'
        flagdf = self.df[[self.fluxcol, scf]].copy()
        flagdf[flagname] = np.nan

        print(f"Calculating {flagname} from {scf} with "
              f"flag 0 (good values) where {scf} < {thres_good}, "
              f"flag 1 (ok values) where {scf} >= {thres_good} and < {thres_ok}, "
              f"flag 2 (bad values) where {scf} >= {thres_ok}...")
        # percentiles_df = percentiles(series=flagdf[scf], showplot=False)

        good = flagdf[scf] < thres_good
        ok = (flagdf[scf] >= thres_good) & (flagdf[scf] < thres_ok)
        bad = flagdf[scf] >= thres_ok

        flagdf[flagname][good] = 0
        flagdf[flagname][ok] = 1
        flagdf[flagname][bad] = 2

        self._fluxflags[flagname] = flagdf[flagname].copy()

    def missing_vals_test(self):
        flagname = f'FLAG_{self.fluxcol}_MISSING_TEST'

        flagdf = self.df[[self.fluxcol]].copy()
        flagdf[flagname] = np.nan

        print(f"Calculating {flagname} from {self.fluxcol} "
              f"with flag 0 (good values) where {self.fluxcol} is available, "
              f"flag 2 (bad values) where {self.fluxcol} is missing ...")
        # percentiles_df = percentiles(series=flagdf[self.flux])

        bad = flagdf[self.fluxcol].isnull()
        good = ~bad

        flagdf[flagname][good] = 0
        flagdf[flagname][bad] = 2

        self._fluxflags[flagname] = flagdf[flagname].copy()

    def ssitc_test(self):
        flagname = f'FLAG_{self.fluxcol}_SSITC_TEST'
        _flagname = f'{self.fluxcol}_SSITC_TEST'  # Name in EddyPro file

        flagdf = self.df[[self.fluxcol, _flagname]].copy()

        print(f"Fetching {_flagname} directly from output file, renaming to {flagname} ...")
        # percentiles_df = percentiles(series=flagdf[flagname])

        self._fluxflags[flagname] = flagdf[_flagname].copy()

    def completeness_test(self,
                          thres_good: float = 0.99,
                          thres_ok: float = 0.97):

        flagname = f'FLAG_{self.fluxcol}_COMPLETENESS_TEST'
        expected_n_records = 'EXPECT_NR'
        gas_n_records = f'{self.gas}_NR'
        gas_n_records_perc = f'{self.gas}_NR_PERC'

        flagdf = self.df[[self.fluxcol]].copy()
        flagdf[expected_n_records] = self.df[expected_n_records].copy()
        flagdf[gas_n_records] = self.df[gas_n_records].copy()
        flagdf[gas_n_records_perc] = flagdf[gas_n_records].divide(flagdf[expected_n_records])
        flagdf[flagname] = np.nan

        print(f"Calculating {flagname} from {gas_n_records_perc} with "
              f"flag 0 (good values) where {gas_n_records_perc} >= {thres_good}, "
              f"flag 1 (ok values) where {gas_n_records_perc} >= {thres_ok} and < {thres_good}, "
              f"flag 2 (bad values) < {thres_ok}...")
        # percentiles_df = percentiles(series=flagdf[gas_n_records_perc], showplot=False)

        good = flagdf[gas_n_records_perc] >= thres_good
        ok = (flagdf[gas_n_records_perc] >= thres_ok) & (flagdf[gas_n_records_perc] < thres_good)
        bad = flagdf[gas_n_records_perc] < thres_ok

        flagdf[flagname][good] = 0
        flagdf[flagname][ok] = 1
        flagdf[flagname][bad] = 2

        self._fluxflags[flagname] = flagdf[flagname].copy()

    def _detect_gas(self) -> str:
        """Detect name of gas that was used to calculate the flux"""
        gas = None
        if self.fluxcol == 'FC':
            gas = 'CO2'
        elif (self.fluxcol == 'FH2O') \
                | (self.fluxcol == 'LE') \
                | (self.fluxcol == 'ET'):
            gas = 'H2O'
        elif (self.fluxcol == 'FN2O'):
            gas = 'N2O'
        elif (self.fluxcol == 'FCH4'):
            gas = 'CH4'
        print(f"Detected gas: {gas}")
        return gas

    def showplot(self, maxflux:float):

        fig = plt.figure(facecolor='white', figsize=(19, 9))
        gs = gridspec.GridSpec(1, 4)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1], sharey=ax_before)
        ax_flagsum = fig.add_subplot(gs[0, 2], sharey=ax_before)
        ax_flag = fig.add_subplot(gs[0, 3], sharey=ax_before)

        HeatmapDateTime(ax=ax_before, series=self.fluxqcf['FC'], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_after, series=self.fluxqcf['FC_QCF'], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flagsum, series=self.fluxflags['SUM_FLAGS'],
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flag, series=self.fluxqcf['FLAG_FC_QCF'],
                        cb_digits_after_comma=0).plot()

        plt.setp(ax_after.get_yticklabels(), visible=False)
        plt.setp(ax_flagsum.get_yticklabels(), visible=False)
        plt.setp(ax_flag.get_yticklabels(), visible=False)

        ax_after.axes.get_yaxis().get_label().set_visible(False)
        ax_flagsum.axes.get_yaxis().get_label().set_visible(False)
        ax_flag.axes.get_yaxis().get_label().set_visible(False)

        # plt.setp(ax_after.get_ylabel(), visible=False)
        # plt.setp(ax_flagsum.get_ylabel(), visible=False)
        # plt.setp(ax_flag.get_ylabel(), visible=False)

        fig.show()


def example():
    # # Load data from files
    # import os
    # from pathlib import Path
    # from diive.core.io.filereader import MultiDataFileReader
    # from diive.core.io.files import save_as_pickle
    # FOLDER = r"F:\Dropbox\luhk_work\20 - CODING\26 - NOTEBOOKS\gl-notebooks\__indev__\data\fluxnet"
    # filepaths = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
    # filepaths = [FOLDER + "/" + f for f in filepaths]
    # filepaths = [Path(f) for f in filepaths]
    # [print(f) for f in filepaths]
    # loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    # df = loaddatafile.data_df
    # save_as_pickle(outpath=r'F:\Dropbox\luhk_work\_temp', filename="data", data=df)

    # Load data from pickle (much faster loading)
    from diive.core.io.files import load_pickle
    df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data.pickle")
    # print(df)
    # [print(c) for c in df.columns if "CUSTOM" in c]

    # FLUXES = ['FC', 'H2O', 'LE', 'ET', 'FCH4', 'FN2O', 'TAU']

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=df['FC'], vmin=-20, vmax=20).show()

    fluxqc = FluxQCF(fluxcol='FC', df=df)
    fluxqc.missing_vals_test()
    fluxqc.ssitc_test()
    fluxqc.completeness_test()
    fluxqc.spectral_correction_factor_test()
    fluxqc.signal_strength_test(signal_strength_col='CUSTOM_AGC_MEAN',
                                method='discard above', threshold=90)
    fluxqc.raw_data_screening_vm97()

    fluxqc.calculate_qcf()

    fluxqc.report_flags()
    fluxqc.report_qcf_evolution()
    fluxqc.report_flux()

    # fluxqc.fluxflags
    # fluxqc.fluxqcf

    fluxqc.showplot(maxflux=25)


if __name__ == '__main__':
    example()
