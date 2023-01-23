import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.flux.common import detect_fluxgas


class StorageCorrectionSinglePoint:

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str):
        self.df = df.copy()
        self.fluxcol = fluxcol

        self.gascol = detect_fluxgas(fluxcol=self.fluxcol)
        self.strgcol = self._detect_storage_var()
        self.flux_corrected_col = self._output_name()
        self.flagname = f'FLAG_L3.1_{self.fluxcol}_{self.strgcol}-MISSING_TEST'

        self._storage = None

    @property
    def storage(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._storage, DataFrame):
            raise Exception('Results for storage are empty')
        return self._storage

    def get(self) -> DataFrame:
        """Return original data with storage-corrected flux and the flag
        indicating storage term availability"""
        _df = self.df.copy()
        _df[self.flux_corrected_col] = self.storage[self.flux_corrected_col].copy()
        _df[self.flagname] = self.storage[self.flagname].copy()
        return _df

    def report(self):
        print(f"\n{'=' * 40}\nREPORT: STORAGE CORRECTION FOR {self.fluxcol}\n{'=' * 40}")
        print(f"Swiss FluxNet processing chain, Level-3.1: Storage Correction")

        stats = self.storage.describe()

        before = int(stats[self.fluxcol]['count'])
        after = int(stats[self.flux_corrected_col]['count'])
        missed = int(before - after)
        n_storageterm = self.storage[self.flagname].loc[self.storage[self.flagname] == 0].count()
        n_storageterm_check = len(self.storage[self.strgcol].dropna())

        print(f"The storage term {self.strgcol} was added to flux {self.fluxcol}.")
        print(f"The storage-corrected flux was stored as {self.flux_corrected_col}.")
        print(f"Before storage correction: {before} records ({self.fluxcol})")
        print(f"After storage correction: {after} records ({self.flux_corrected_col})")
        print(f"The storage term is available for: {n_storageterm} records ({self.flagname})")
        print(f"The storage term is available for: {n_storageterm_check} records ({self.strgcol})")

        if missed > 0:
            print(f"Storage-corrected {self.flux_corrected_col} could not be calculated for "
                  f"a total of {missed} flux records because no storage term was available "
                  f"for these records.")

        print(stats)

    def apply_storage_correction(self):
        print(f"Calculating storage-corrected flux {self.flux_corrected_col} "
              f"from flux {self.fluxcol} and storage term {self.strgcol} ...")
        self._storage = self.df[[self.fluxcol, self.strgcol]].copy()
        self._storage[self.flux_corrected_col] = self._storage[self.fluxcol].add(self._storage[self.strgcol])
        self._storageterm_missing_test()

    def _storageterm_missing_test(self):
        """Add flag that shows if the storage term is available"""
        self._storage[self.flagname] = np.nan
        bad = self._storage[self.strgcol].isnull()
        good = ~bad
        self._storage[self.flagname][good] = 0
        self._storage[self.flagname][bad] = 2

    def _output_name(self) -> str:
        """
        Variable name for the storage-corrected flux

        Storage-corrected fluxes get the suffix L3.1 because adding
        the storage term is Level-3.1 in the Swiss FluxNet processing
        chain.
        """
        if self.fluxcol == 'FC':
            flux_corrected_col = 'NEE_L3.1'
        else:
            flux_corrected_col = f"{self.fluxcol}_L3.1"
        return flux_corrected_col

    def _detect_storage_var(self) -> str:
        """Detect name of gas column that was used to calculate the flux"""
        strgcol = None
        if self.fluxcol == 'FC':
            strgcol = 'SC_SINGLE'
        elif (self.fluxcol == 'FH2O'):
            strgcol = 'SH2O_SINGLE'
        elif (self.fluxcol == 'LE'):
            strgcol = 'SLE_SINGLE'
        elif (self.fluxcol == 'ET'):
            strgcol = 'SET_SINGLE'
        elif (self.fluxcol == 'FN2O'):
            strgcol = 'SN2O_SINGLE'
        elif (self.fluxcol == 'FCH4'):
            strgcol = 'SCH4_SINGLE'
        elif (self.fluxcol == 'H'):
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
    from diive.core.io.files import load_pickle
    df = load_pickle(filepath=r"L:\Sync\luhk_work\_temp\data.pickle")

    s = StorageCorrectionSinglePoint(df=df, fluxcol='FC')
    s.apply_storage_correction()
    # s.showplot(maxflux=20)
    # print(s.storage)
    s.report()

    df = s.get()

    # [print(c) for c in df.columns if "TAU" in c]


if __name__ == '__main__':
    example()
