"""

MARGINAL DISTRIBUTION SAMPLING (MDS)
Gap-filling after Reichstein et al (2005)

Reference: https://doi.org/10.1111/j.1365-2486.2005.001002.x

"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame


class FluxMDS:

    def __init__(self,
                 df: DataFrame,
                 flux: str,
                 swin: str,
                 ta: str,
                 vpd: str,
                 swin_class: float = 50,
                 ta_class: float = 2.5,
                 vpd_class: float = 0.5,
                 verbose: int = 1):
        self._df = df[[flux, swin, ta, vpd]].copy()
        self.flux = flux
        self.swin = swin
        self.ta = ta
        self.vpd = vpd
        self.swin_class = swin_class
        self.ta_class = ta_class
        self.vpd_class = vpd_class
        self.verbose = verbose

        self._add_newcols()

        self.workdf = DataFrame()

    @property
    def df(self) -> DataFrame:
        """Dataframe containing all data."""
        if not isinstance(self._df, DataFrame):
            raise Exception('No overall flag available.')
        return self._df

    def _a_1_2(self, row):
        locs = (
                (self.df.index >= row['.START'])
                & (self.df.index <= row['.END'])
                & (self.df[f'.{self.ta}_UPPERLIM'] > row[self.ta])
                & (self.df[f'.{self.ta}_LOWERLIM'] < row[self.ta])
                & (self.df[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.df[f'.{self.swin}_LOWERLIM'] < row[self.swin])
                & (self.df[f'.{self.vpd}_UPPERLIM'] > row[self.vpd])
                & (self.df[f'.{self.vpd}_LOWERLIM'] < row[self.vpd])
        )
        _array = self.df.loc[locs, self.flux].to_numpy()
        n_vals = len(_array[~np.isnan(_array)])
        if n_vals > 0:
            avg = np.nanmean(_array)
        else:
            avg = np.nan

        return avg

    def _a3(self, row):
        locs = (
                (self.df.index >= row['.START'])
                & (self.df.index <= row['.END'])
                & (self.df[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.df[f'.{self.swin}_LOWERLIM'] < row[self.swin])
        )
        _array = self.df.loc[locs, self.flux].to_numpy()
        n_vals = len(_array[~np.isnan(_array)])
        if n_vals > 0:
            avg = np.nanmean(_array)
        else:
            avg = np.nan

        return avg

    def _fill_predictions(self, _df, workdf) -> DataFrame:
        _df['.PREDICTIONS'] = _df['.PREDICTIONS'].fillna(workdf['.PREDICTIONS'])
        _df['.START'] = _df['.START'].fillna(pd.to_datetime(workdf['.START'])).infer_objects(copy=False)
        _df['.END'] = _df['.END'].fillna(workdf['.END'])
        _df['.PREDICTIONS_QUALITY'] = _df['.PREDICTIONS_QUALITY'].fillna(workdf['.PREDICTIONS_QUALITY'])
        return _df

    def _run_two_available(self, days: int, quality: int):
        print(f"Gap-filling quality {quality} ...")
        _df = self.df.copy()
        workdf = self.workdf.copy()

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        workdf['.PREDICTIONS'] = workdf.apply(self._a3, axis=1)
        workdf['.PREDICTIONS_QUALITY'] = quality

        _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _run_all_available(self, days: int, quality: int):
        print(f"Gap-filling quality {quality} ...")
        _df = self.df.copy()
        workdf = self.workdf.copy()

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        workdf['.PREDICTIONS'] = workdf.apply(self._a_1_2, axis=1)
        workdf['.PREDICTIONS_QUALITY'] = quality

        if quality == 1:
            _df['.PREDICTIONS'] = workdf['.PREDICTIONS'].copy()
            _df['.START'] = workdf['.START'].copy()
            _df['.END'] = workdf['.END'].copy()
            _df['.PREDICTIONS_QUALITY'] = workdf['.PREDICTIONS_QUALITY'].copy()
        else:
            _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def run(self):
        # https://www.geeksforgeeks.org/apply-function-to-every-row-in-a-pandas-dataframe/
        # https://labs.quansight.org/blog/unlocking-c-level-performance-in-df-apply

        self._df = self.df.copy()
        locs_missing = self.df['.PREDICTIONS'].isnull()
        self.workdf = self._df[locs_missing].copy()

        # A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
        self.workdf, self._df = self._run_all_available(days=7, quality=1)

        # A2: SWIN, TA, VPD, NEE available within 14 days
        self.workdf, self._df = self._run_all_available(days=14, quality=2)

        # A3: SWIN, NEE available within 7 days
        self.workdf, self._df = self._run_two_available(days=7, quality=3)

        # A4: NEE available within |dt| <= 1h on same day
        pass

        # B1: same hour NEE available within |dt| <= 1 day
        pass

        # B2: SWIN, TA, VPD, NEE available within 21 days
        self.workdf, self._df = self._run_all_available(days=21, quality=6)

        # B3: SWIN, TA, VPD, NEE available within 28 days
        self.workdf, self._df = self._run_all_available(days=28, quality=7)

        print(len(self.df['.PREDICTIONS']))
        print(self.df['.PREDICTIONS'].sum())
        import matplotlib.pyplot as plt
        self.df['.PREDICTIONS'].plot(label="predictions")
        self.df[self.flux].plot()
        plt.legend()
        plt.show()

    def _add_newcols(self):
        self._df['.TIMESTAMP'] = self.df.index
        self._df['.PREDICTIONS'] = np.nan
        self._df['.PREDICTIONS_SD'] = np.nan
        self._df['.PREDICTIONS_COUNTS'] = np.nan
        self._df['.PREDICTIONS_QUALITY'] = np.nan
        self._df['.START'] = np.nan
        self._df['.END'] = np.nan
        self._df[f'.{self.swin}_LOWERLIM'] = self.df[self.swin].sub(self.swin_class)
        self._df[f'.{self.swin}_UPPERLIM'] = self.df[self.swin].add(self.swin_class)
        self._df[f'.{self.ta}_LOWERLIM'] = self.df[self.ta].sub(self.ta_class)
        self._df[f'.{self.ta}_UPPERLIM'] = self.df[self.ta].add(self.ta_class)
        self._df[f'.{self.vpd}_LOWERLIM'] = self.df[self.vpd].sub(self.vpd_class)
        self._df[f'.{self.vpd}_UPPERLIM'] = self.df[self.vpd].add(self.vpd_class)


def example():
    from diive.core.io.files import load_parquet

    SOURCEDIR = r"L:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2024_2005-2023\40_FLUXES_L1_IRGA+QCL+LGR_mergeData"
    FILENAME = r"41.1_CH-CHA_IRGA_LGR+QCL_Level-1_eddypro_fluxnet_2005-2023_meteo7.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    df = load_parquet(filepath=FILEPATH)

    flux = 'FC'
    ta = 'TA_T1_2_1'
    swin = 'SW_IN_T1_2_1'
    vpd = 'VPD_T1_2_1'
    ustar = 'USTAR'
    ssitc = 'FC_SSITC_TEST'
    swin_class = 50  # W m-2
    ta_class = 2.5  # °C
    vpd_class = 0.5  # kPa; 5 hPa is default for reference

    locs = (df.index.year >= 2021) & (df.index.year <= 2021) & (df.index.month >= 9) & (
            df.index.month <= 9)
    subsetcols = [flux, swin, ta, vpd, ustar, ssitc]
    subsetdf = df.loc[locs, subsetcols].copy()
    good = (subsetdf[flux] > -50) & (subsetdf[flux] < 50) & (subsetdf[ustar] > 0.1) & (subsetdf[ssitc] < 1)
    subsetdf.loc[~good, flux] = np.nan
    # subsetdf.describe()

    import time
    a = time.perf_counter()
    mds = FluxMDS(
        df=subsetdf,
        flux=flux,
        ta=ta,
        swin=swin,
        vpd=vpd,
        swin_class=swin_class,
        ta_class=ta_class,
        vpd_class=vpd_class  # kPa; 5 hPa is default for reference
    )
    mds.run()
    b = time.perf_counter()
    print(f"Duration: {b - a}")


if __name__ == '__main__':
    example()
