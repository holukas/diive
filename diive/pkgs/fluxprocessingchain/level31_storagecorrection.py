import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from diive.core.dfun.stats import sstats  # Time series stats
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.heatmap_datetime import HeatmapDateTime


class FluxStorageCorrectionSinglePointEddyPro:
    """
    Estimation of storage fluxes (gases, sensible heat, latent heat) from concentrations
    (1-point profile) as calculated by EddyPro
    """

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 basevar: str,
                 gapfill_storage_term: bool = False,
                 idstr: str = 'L3.1'):
        self.df = df.copy()
        self.fluxcol = fluxcol
        self.basevar = basevar
        self.gapfill_storage_term = gapfill_storage_term
        self.idstr = validate_id_string(idstr=idstr)
        self.flux_corrected_col, self.strgcol = self._detect_storage_var()
        self.flagname = f'FLAG{self.idstr}_{self.fluxcol}_{self.strgcol}-MISSING_TEST'

        # Name of gapfilled storage column and its flag
        self.gapfilled_strgcol = None
        # self.gapfilled_strgcol = f"{self.strgcol}_gfRF{self.idstr}"
        self.flag_isgapfilled = None
        # self.flag_isgapfilled = f"FLAG_{self.gapfilled_strgcol}_ISFILLED"

        self._results = None

    @property
    def results(self) -> DataFrame:
        """Return results as dataframe"""
        if not isinstance(self._results, DataFrame):
            raise Exception('Results for storage are empty')
        return self._results

    def report(self):
        print(f"\n{'=' * 40}\nREPORT: STORAGE CORRECTION FOR {self.fluxcol}\n{'=' * 40}")
        print(f"Swiss FluxNet processing chain, {self.idstr}: Storage Correction")

        print(f"\nThe gap-filled storage term {self.gapfilled_strgcol} was added to flux {self.fluxcol}.")
        print(f"The storage-corrected flux was stored as {self.flux_corrected_col}.")

        n_flux = len(self.results[self.fluxcol].dropna())
        print(f"\nThe flux was available for {n_flux} records ({self.fluxcol}).")

        n_storageterm = len(self.results[self.strgcol].dropna())
        print(f"The original, non-gapfilled storage term was available for "
              f"{n_storageterm} records ({self.strgcol}).")

        # Generate temporary subset where all flux values are available and check for missing storage
        _subset = pd.concat([self.results[self.fluxcol], self.results[self.strgcol]], axis=1)
        _subset = _subset.dropna(subset=[self.fluxcol])
        n_orig_missing_strg = _subset[self.strgcol].isnull().sum()
        print(f"The original storage term {self.strgcol} was missing for {n_orig_missing_strg} "
              f"flux records.")
        print(f"Without gap-filling the storage term ({self.strgcol}), "
              f"{n_orig_missing_strg} measured flux records ({self.fluxcol}) are lost.")

        if (n_storageterm > n_flux) & (n_orig_missing_strg > 0):
            print(f"NOTE: There were more values available for storage term {self.strgcol} "
                  f"than for flux {self.fluxcol}.\n"
                  f"However, for {n_orig_missing_strg} flux records "
                  f"no concurrent storage terms were available.")

        if self.gapfilled_strgcol:
            print(f"\nFor this run, gap-filling of {self.strgcol} was * SELECTED *.")

            locs_fluxmissing = self.results[self.fluxcol].isnull()
            fluxavailable = self.results[~locs_fluxmissing].copy()
            locs_isfilled = fluxavailable[self.flag_isgapfilled] == 1
            locs_isorig = fluxavailable[self.flag_isgapfilled] == 0
            n_isfilled = len(fluxavailable[locs_isfilled])  # Filled storage terms for available fluxes
            n_isorig = len(fluxavailable[locs_isorig])  # Filled storage terms for available fluxes
            print(f"After gap-filling the storage term, it was available for an additional "
                  f"{n_isfilled} records ({self.gapfilled_strgcol}).")

            perc1 = (n_isorig / n_flux) * 100
            perc2 = (n_orig_missing_strg / n_flux) * 100
            n_flux_corrected = self.results[self.flux_corrected_col].dropna().count()
            print(f"\nIn the storage-corrected flux {self.flux_corrected_col} with {n_flux_corrected} records, "
                  f"\n  - {perc1:.1f}% ({n_storageterm} records) of used storage terms come from originally calculated data ({self.strgcol})"
                  f"\n  - {perc2:.1f}% ({n_orig_missing_strg} records) of used storage terms come from gap-filled data ({self.gapfilled_strgcol})")

            filledstats = sstats(fluxavailable[locs_isfilled][self.gapfilled_strgcol])
            print(f"\nStats for gap-filled storage terms:"
                  f"\n{filledstats.T[['NOV', 'P01', 'MEDIAN', 'P99']]}")

            measuredstats = sstats(fluxavailable[~locs_isfilled][self.gapfilled_strgcol])
            print(f"\nStats for measured storage terms:"
                  f"\n{measuredstats.T[['NOV', 'P01', 'MEDIAN', 'P99']]}")

        else:
            print(f"\nFor this run, gap-filling of {self.strgcol} was - NOT SELECTED -.")

    def _gapfill_storage_term(self) -> DataFrame:
        """Gap-fill storage term using rolling mean in expanding window.

        The storage term can be missing for quite a few records,
        which means that we lose measured flux data.
        """

        # New columns
        self.gapfilled_strgcol = f"{self.strgcol}_gfRMED{self.idstr}"
        self.flag_isgapfilled = f"FLAG_{self.gapfilled_strgcol}_ISFILLED"

        # Generate temporary subset where all flux values are available and check for missing storage
        gapfilled_df = pd.concat([self.results[self.fluxcol], self.results[self.strgcol]], axis=1)
        gapfilled_df[self.gapfilled_strgcol] = self.results[self.strgcol].copy()
        gapfilled_df[self.flag_isgapfilled] = 0
        gapfilled_df = gapfilled_df.dropna(subset=[self.fluxcol])
        n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
        prev_n_still_missing_strg = n_still_missing_strg
        print(f"Missing values for storage term {self.gapfilled_strgcol}: {n_still_missing_strg}")

        # Fill gaps with rolling mean in expanding time window
        window_size = 0
        while n_still_missing_strg > 0:
            window_size = 3 if window_size == 0 else window_size + 2
            rmedian = self.results[self.strgcol].rolling(window=window_size, center=True, min_periods=3).median()
            locs = gapfilled_df[self.gapfilled_strgcol].isnull()
            gapfilled_df.loc[locs, self.gapfilled_strgcol] = rmedian
            gapfilled_df.loc[locs, self.flag_isgapfilled] = 1
            n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
            if n_still_missing_strg < prev_n_still_missing_strg:
                print(f"Gap-filling storage-term {self.strgcol} "
                      f"with rolling median (window size = {window_size} records) "
                      f"Still missing values for storage term {self.gapfilled_strgcol}: {n_still_missing_strg} ...")
                prev_n_still_missing_strg = n_still_missing_strg

        gapfilled_df = gapfilled_df[[self.gapfilled_strgcol, self.flag_isgapfilled]].copy()

        return gapfilled_df

    def storage_correction(self):
        print(f"Calculating storage-corrected flux {self.flux_corrected_col} "
              f"from flux {self.fluxcol} and storage term {self.strgcol} ...")

        # Collect flux and storage term data
        self._results = self.df[[self.fluxcol, self.strgcol]].copy()

        # Gap-fill storage term
        if self.gapfill_storage_term:
            gapfilled_df = self._gapfill_storage_term()
            self._results = pd.concat([self._results, gapfilled_df], axis=1)

            # Add gapfilled storage term to flux data
            self._results[self.flux_corrected_col] = self._results[self.fluxcol].add(
                self._results[self.gapfilled_strgcol])
        else:
            # Add original (non-gapfilled) storage term to flux, this can result in NaNs
            self._results[self.flux_corrected_col] = self._results[self.fluxcol].add(self._results[self.strgcol])

    def _detect_storage_var(self) -> tuple[str, str]:
        """Detect name of gas column that was used to calculate the flux, and set
        variable name for the storage-corrected flux.

        Storage-corrected fluxes get the suffix L3.1 because adding
        the storage term is Level-3.1 in the Swiss FluxNet processing
        chain.
        """

        flux_corrected_col = None

        options = {
            'FC': 'SC_SINGLE',
            'FH2O': 'SH2O_SINGLE',
            'LE': 'SLE_SINGLE',
            'ET': 'SET_SINGLE',
            'FN2O': 'SN2O_SINGLE',
            'FCH4': 'SCH4_SINGLE',
            'H': 'SH_SINGLE'
        }
        if self.fluxcol == 'FC':
            flux_corrected_col = f'NEE{self.idstr}'

        # elif self.filetype == 'EDDYPRO-FULL-OUTPUT-CSV-30MIN':
        #     options = {
        #         'co2_flux': 'co2_strg',
        #         'h2o_flux': 'h2o_strg',
        #         'LE': 'LE_strg',
        #         'n2o_flux': 'n2o_strg',
        #         'ch4_flux': 'ch4_strg',
        #         'H': 'H_strg'
        #     }
        #     if self.fluxcol == 'co2_flux':
        #         flux_corrected_col = f'NEE{self.idstr}'

        strgcol = options[self.fluxcol]
        if not flux_corrected_col:
            flux_corrected_col = f"{self.fluxcol}{self.idstr}"
        print(f"Detected storage variable {strgcol} for {self.fluxcol}.")
        return flux_corrected_col, strgcol

    def showplot(self, maxflux: float = None):

        fig = plt.figure(facecolor='white', figsize=(20, 9))
        gs = gridspec.GridSpec(1, 5)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_flux = fig.add_subplot(gs[0, 0])
        ax_flux_storage_corrected = fig.add_subplot(gs[0, 1], sharey=ax_flux)
        ax_storage_term = fig.add_subplot(gs[0, 2], sharey=ax_flux)
        ax_storage_term_gf = fig.add_subplot(gs[0, 3], sharey=ax_flux)
        ax_storage_term_flag = fig.add_subplot(gs[0, 4], sharey=ax_flux)

        if not maxflux:
            maxflux = self.results[self.fluxcol].abs().quantile(.95)
            maxstrg = self.results[self.strgcol].abs().quantile(.95)
        else:
            maxstrg = maxflux

        HeatmapDateTime(ax=ax_flux, series=self.results[self.fluxcol], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flux_storage_corrected, series=self.results[self.flux_corrected_col], vmin=-maxflux,
                        vmax=maxflux, cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_storage_term, series=self.results[self.strgcol], vmin=-maxstrg, vmax=maxstrg,
                        cb_digits_after_comma=1).plot()
        HeatmapDateTime(ax=ax_storage_term_gf, series=self.results[self.gapfilled_strgcol], vmin=-maxstrg, vmax=maxstrg,
                        cb_digits_after_comma=1).plot()
        HeatmapDateTime(ax=ax_storage_term_flag, series=self.results[self.flag_isgapfilled],
                        cb_digits_after_comma=0).plot()

        plt.setp(ax_flux_storage_corrected.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term_gf.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term_flag.get_yticklabels(), visible=False)

        ax_flux_storage_corrected.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term_gf.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()


def example():
    pass
    # # Load data from pickle (much faster loading)
    # from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    # df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    # s = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC')
    # s.storage_correction()
    # # s.showplot(maxflux=20)
    # # print(s.storage)
    # s.report()
    # # df = s.addresults()
    # # [print(c) for c in df.columns if "TAU" in c]


if __name__ == '__main__':
    example()
