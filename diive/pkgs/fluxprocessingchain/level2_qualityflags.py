import numpy as np
import pandas as pd
from pandas import DataFrame

from diive.pkgs.flux.common import detect_fluxgas
from pkgs.qaqc.qcf import FluxQCF


class QualityFlagsLevel2:

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str):
        """
        Create QCF (quality-control flag) for selected flags, calculated
        from EddyPro's _fluxnet_ output files

        Args:
            df: Dataframe containing data from EddyPro's _fluxnet_ file
            fluxcol: Name of the flux variable in *df*
        """
        self.fluxcol = fluxcol
        self.df = df.copy()

        self.gas = detect_fluxgas(fluxcol=self.fluxcol)

        self._fluxflags = self.df[[fluxcol]].copy()

    @property
    def fluxflags(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._fluxflags, DataFrame):
            raise Exception('Results for flux flags are empty')
        return self._fluxflags

    def get(self) -> DataFrame:
        """Return original data with QCF flag"""
        _df = self.df.copy()  # Main data
        test_cols = [t for t in self.fluxflags.columns if str(t).startswith('FLAG_L2_')]
        flags_df = self.fluxflags[test_cols].copy()  # Subset w/ flags
        [print(f"++Adding new column {c} (Level-2 quality flag) to main data ...") for c in test_cols]
        _df = pd.concat([_df, flags_df], axis=1)  # Add flags to main data
        return _df

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
            '1': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_SPIKE_HF_TEST",  # Spike detection, hard flag
            '2': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_AMPLITUDE_RESOLUTION_HF_TEST",
            # Amplitude resolution, hard flag
            '3': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_DROPOUT_TEST",  # Drop-out, hard flag
            '4': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_ABSOLUTE_LIMITS_HF_TEST",  # Absolute limits, hard flag
            '5': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_SKEWKURT_HF_TEST",  # Skewness/kurtosis, hard flag
            '6': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_SKEWKURT_SF_TEST",  # Skewness/kurtosis, soft flag
            '7': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_DISCONTINUITIES_HF_TEST",  # Discontinuities, hard flag
            '8': f"FLAG_L2_{self.fluxcol}_{self.gas}_VM97_DISCONTINUITIES_SF_TEST"  # Discontinuities, soft flag
        }

        # Extract 8 individual flags from VM97 multi-flag integer
        for i, c in flagcols.items():
            flagdf[c] = flagdf[vm97].astype(str).str[int(i)].astype(float) \
                .replace(9, np.nan)
            if '_HF_' in c:
                # Hard flag 1 corresponds to bad value
                flagdf[c] = flagdf[c].replace(1, 2)

        # Make new dict that contains flags that we use later
        flagcols_used = {x: flagcols[x] for x in flagcols if x in ('1', '3')}
        # flagcols_used = {x: flagcols[x] for x in flagcols if x in ('1', '3', '4')}

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
        flagname = f'FLAG_L2_{self.fluxcol}_SIGNAL_STRENGTH_TEST'
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
        flagname = f'FLAG_L2_{self.fluxcol}_SCF_TEST'
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
        flagname = f'FLAG_L2_{self.fluxcol}_MISSING_TEST'

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
        flagname = f'FLAG_L2_{self.fluxcol}_SSITC_TEST'
        _flagname = f'{self.fluxcol}_SSITC_TEST'  # Name in EddyPro file

        flagdf = self.df[[self.fluxcol, _flagname]].copy()

        print(f"Fetching {_flagname} directly from output file, renaming to {flagname} ...")
        # percentiles_df = percentiles(series=flagdf[flagname])

        self._fluxflags[flagname] = flagdf[_flagname].copy()

    def gas_completeness_test(self,
                              thres_good: float = 0.99,
                              thres_ok: float = 0.97):

        flagname = f'FLAG_L2_{self.fluxcol}_COMPLETENESS_TEST'
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


def example():
    # # Load data from files
    # import os
    # from pathlib import Path
    # from diive.core.io.filereader import MultiDataFileReader
    # from diive.core.io.files import save_as_pickle
    # FOLDER = r"L:\Sync\luhk_work\20 - CODING\26 - NOTEBOOKS\gl-notebooks\__indev__\data\fluxnet"
    # filepaths = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
    # filepaths = [FOLDER + "/" + f for f in filepaths]
    # filepaths = [Path(f) for f in filepaths]
    # [print(f) for f in filepaths]
    # loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    # df = loaddatafile.data_df
    # save_as_pickle(outpath=r'F:\Dropbox\luhk_work\_temp', filename="data", data=df)

    # Load data from pickle (much faster loading)
    from diive.core.io.files import load_pickle
    df = load_pickle(filepath=r"L:\Sync\luhk_work\_temp\data.pickle")
    # print(df)
    # [print(c) for c in df.columns if "CUSTOM" in c]

    # FLUXES = ['FC', 'H2O', 'LE', 'ET', 'FCH4', 'FN2O', 'TAU']

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=df['FC'], vmin=-20, vmax=20).show()

    fluxqc = QualityFlagsLevel2(fluxcol='FC', df=df)
    fluxqc.missing_vals_test()
    fluxqc.ssitc_test()
    fluxqc.gas_completeness_test()
    fluxqc.spectral_correction_factor_test()
    fluxqc.signal_strength_test(signal_strength_col='CUSTOM_AGC_MEAN',
                                method='discard above', threshold=90)
    fluxqc.raw_data_screening_vm97()
    print(fluxqc.fluxflags)
    _df = fluxqc.get()

    qcf = FluxQCF(df=_df, fluxcol='FC', level=2, swinpotcol='SW_IN_POT',
                  nighttime_threshold=50,
                  daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=1)
    qcf.calculate()
    qcf.report_flags()
    qcf.report_qcf_evolution()
    qcf.report_flux()
    qcf.showplot(maxflux=10)
    _df = qcf.get()


if __name__ == '__main__':
    example()
