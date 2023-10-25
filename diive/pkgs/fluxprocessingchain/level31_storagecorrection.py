import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from diive.core.dfun.stats import sstats  # Time series stats
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.flux.common import detect_flux_basevar
from diive.pkgs.gapfilling.randomforest_ts import QuickFillRFTS


class FluxStorageCorrectionSinglePointEddyPro:
    """
    Estimation of storage fluxes (gases, sensible heat, latent heat) from concentrations
    (1-point profile) as calculated by EddyPro
    """

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 levelid: str = 'L3.1'):
        self.df = df.copy()
        self.fluxcol = fluxcol
        self.levelid = levelid if levelid else ""

        self.gascol = detect_flux_basevar(fluxcol=self.fluxcol)
        self.strgcol = self._detect_storage_var()
        self.flux_corrected_col = self._output_name()
        self.flagname = f'FLAG_{self.levelid}_{self.fluxcol}_{self.strgcol}-MISSING_TEST'

        # Name of gapfilled storage column and its flag
        self.gapfilled_strgcol = f"{self.strgcol}_gfRF_{self.levelid}"
        self.flag_isgapfilled = f"FLAG_{self.gapfilled_strgcol}_ISFILLED"

        self._storage = None

    @property
    def storage(self) -> DataFrame:
        """Return full dataframe including storage corrected columns"""
        if not isinstance(self._storage, DataFrame):
            raise Exception('Results for storage are empty')
        return self._storage

    def get(self) -> DataFrame:
        """Return original data with storage-corrected flux and the flag
        indicating storage term availability"""
        df = self.df.copy()  # Main data
        return_cols = [self.flux_corrected_col, self.gapfilled_strgcol]
        [print(f"++Adding new column {c} to main data ...") for c in return_cols]
        storage_df = self.storage[return_cols].copy()
        df = pd.concat([df, storage_df], axis=1)  # Add storage columns to main data
        return df

    def report(self):
        print(f"\n{'=' * 40}\nREPORT: STORAGE CORRECTION FOR {self.fluxcol}\n{'=' * 40}")
        print(f"Swiss FluxNet processing chain, {self.levelid}: Storage Correction")

        print(f"\nThe gap-filled storage term {self.gapfilled_strgcol} was added to flux {self.fluxcol}.")
        print(f"The storage-corrected flux was stored as {self.flux_corrected_col}.")

        n_flux = len(self.storage[self.fluxcol].dropna())
        print(f"\nThe flux was available for {n_flux} records ({self.fluxcol}).")

        n_storageterm = len(self.storage[self.strgcol].dropna())
        print(f"Originally, the non-gapfilled storage term was available for "
              f"{n_storageterm} records ({self.strgcol}).")

        n_missing = n_flux - n_storageterm
        print(f"This means that the storage term {self.strgcol} was missing for "
              f"{n_missing} measured flux ({self.fluxcol}) records.")

        locs_fluxmissing = self.storage[self.fluxcol].isnull()
        fluxavailable = self.storage[~locs_fluxmissing].copy()
        locs_isfilled = fluxavailable[self.flag_isgapfilled] == 1
        n_isfilled = len(fluxavailable[locs_isfilled])  # Filled storage terms for available fluxes
        print(f"After gap-filling the storage term, it was available for an additional "
              f"{n_isfilled} records ({self.gapfilled_strgcol}).")

        perc1 = (n_storageterm / n_flux) * 100
        perc2 = (n_missing / n_flux) * 100
        n_flux_corrected = self.storage[self.flux_corrected_col].dropna().count()
        print(f"\nIn the storage-corrected flux {self.flux_corrected_col} with {n_flux_corrected} records, "
              f"\n  - {perc1:.1f}% ({n_storageterm} records) of used storage terms come from originally calculated data ({self.strgcol})"
              f"\n  - {perc2:.1f}% ({n_missing} records) of used storage terms come from gap-filled data ({self.gapfilled_strgcol})")

        filledstats = sstats(fluxavailable[locs_isfilled][self.gapfilled_strgcol])
        print(f"\nStats for gap-filled storage terms:"
              f"\n{filledstats.T[['NOV', 'P01', 'MEDIAN', 'P99']]}")

        measuredstats = sstats(fluxavailable[~locs_isfilled][self.gapfilled_strgcol])
        print(f"\nStats for measured storage terms:"
              f"\n{measuredstats.T[['NOV', 'P01', 'MEDIAN', 'P99']]}")

    def _gapfill_storage_term(self) -> DataFrame:
        """
        Gap-fill storage term using random forest

        The storage term can be missing for quite a few records,
        which means that we lose measured flux data.
        """
        # Assemble dataframe for gapfilling
        gfcols = [self.strgcol]
        gf_df = self.df[gfcols].copy()

        # Run random forest
        qf = QuickFillRFTS(df=gf_df, target_col=self.strgcol)
        qf.fill()
        print(qf.report())
        series = qf.get_gapfilled_target()
        flag = qf.get_flag()
        d = {series.name: series, flag.name: flag}
        gapfilled_df = pd.DataFrame.from_dict(d)

        # Add Level-ID to gapfilling results
        renamedcols = [f"{c}_{self.levelid}" for c in gapfilled_df.columns]
        gapfilled_df.columns = renamedcols
        self.gapfilled_strgcol = f"{series.name}_{self.levelid}"
        self.flag_isgapfilled = f"{flag.name}_{self.levelid}"
        return gapfilled_df

    def storage_correction(self):
        print(f"Calculating storage-corrected flux {self.flux_corrected_col} "
              f"from flux {self.fluxcol} and storage term {self.strgcol} ...")

        # Gapfill storage term
        subset = self._gapfill_storage_term()

        # Collect original (flux, storage term), gapfilled (storage term) and
        # calculated (storage-corrected flux) data in dataframe
        self._storage = self.df[[self.fluxcol, self.strgcol]].copy()
        self._storage = pd.concat([self._storage, subset], axis=1)

        # Add gapfilled storage data to flux data
        self._storage[self.flux_corrected_col] = self._storage[self.fluxcol].add(self._storage[self.gapfilled_strgcol])
        # self._storageterm_missing_test()

    # def _storageterm_missing_test(self):
    #     """Add flag that shows if the storage term is available"""
    #     self._storage[self.flagname] = np.nan
    #     bad = self._storage[self.strgcol].isnull()
    #     good = ~bad
    #     self._storage[self.flagname][good] = 0
    #     self._storage[self.flagname][bad] = 2

    def _output_name(self) -> str:
        """
        Variable name for the storage-corrected flux

        Storage-corrected fluxes get the suffix L3.1 because adding
        the storage term is Level-3.1 in the Swiss FluxNet processing
        chain.
        """
        if self.fluxcol == 'FC':
            flux_corrected_col = f'NEE_{self.levelid}'
        else:
            flux_corrected_col = f"{self.fluxcol}_{self.levelid}"
        return flux_corrected_col

    def _detect_storage_var(self) -> str:
        """Detect name of gas column that was used to calculate the flux"""
        strgcol = None
        if self.fluxcol == 'FC':
            strgcol = 'SC_SINGLE'
        elif self.fluxcol == 'FH2O':
            strgcol = 'SH2O_SINGLE'
        elif self.fluxcol == 'LE':
            strgcol = 'SLE_SINGLE'
        elif self.fluxcol == 'ET':
            strgcol = 'SET_SINGLE'
        elif self.fluxcol == 'FN2O':
            strgcol = 'SN2O_SINGLE'
        elif self.fluxcol == 'FCH4':
            strgcol = 'SCH4_SINGLE'
        elif self.fluxcol == 'H':
            strgcol = 'SH_SINGLE'
        print(f"Detected storage variable {strgcol} for {self.fluxcol}.")
        return strgcol

    def showplot(self, maxflux: float):

        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 3)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_flux = fig.add_subplot(gs[0, 0])
        ax_flux_storage_corrected = fig.add_subplot(gs[0, 1], sharey=ax_flux)
        ax_storage_term = fig.add_subplot(gs[0, 2], sharey=ax_flux)

        HeatmapDateTime(ax=ax_flux, series=self.storage[self.fluxcol], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flux_storage_corrected, series=self.storage[self.flux_corrected_col], vmin=-maxflux,
                        vmax=maxflux, cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_storage_term, series=self.storage[self.strgcol], vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0).plot()

        plt.setp(ax_flux_storage_corrected.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term.get_yticklabels(), visible=False)

        ax_flux_storage_corrected.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term.axes.get_yaxis().get_label().set_visible(False)

        fig.show()


def example():
    # Load data from pickle (much faster loading)
    from diive.configs.exampledata import load_exampledata_eddypro_fluxnet_CSV_30MIN
    df, _ = load_exampledata_eddypro_fluxnet_CSV_30MIN()
    s = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC')
    s.storage_correction()
    # s.showplot(maxflux=20)
    # print(s.storage)
    s.report()

    df = s.get()

    # [print(c) for c in df.columns if "TAU" in c]


if __name__ == '__main__':
    example()
