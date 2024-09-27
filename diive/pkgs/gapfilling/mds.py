"""

MARGINAL DISTRIBUTION SAMPLING (MDS)
Gap-filling after Reichstein et al (2005)

Reference: https://doi.org/10.1111/j.1365-2486.2005.001002.x

"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

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

    @property
    def df(self) -> DataFrame:
        """Dataframe containing all data."""
        if not isinstance(self._df, DataFrame):
            raise Exception('No overall flag available.')
        return self._df

    def _a1(self, row):
        """A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
        Select rows for gap-filling a specific record.
        """
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
        avg = np.nanmean(self.df.loc[locs, self.flux].to_numpy())
        return avg

    def _run_a1(self):
        offset = pd.DateOffset(days=7)
        self._df['.START'] = pd.to_datetime(self.df.index) - offset
        self._df['.END'] = pd.to_datetime(self.df.index) + offset
        self._df['.PREDICTIONS'] = self.df.apply(self._a1, axis=1)
        self._df['.PREDICTIONS_QUALITY'] = 1
        return self.df


    def run(self):
        # https://www.geeksforgeeks.org/apply-function-to-every-row-in-a-pandas-dataframe/
        # https://labs.quansight.org/blog/unlocking-c-level-performance-in-df-apply

        self._df = self._run_a1()

        print(len(self.df['.PREDICTIONS']))
        print(self.df['.PREDICTIONS'])
        self.df['.PREDICTIONS'].plot(label="predictions")
        self.df[self.flux].plot()
        plt.legend()
        plt.show()

    # def _a1(self, row) -> tuple[bool, int]:
    #     """A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
    #     Select rows for gap-filling a specific record.
    #     """
    #     quality = 1  # 1 = Highest gap-filling quality
    #     offset = pd.DateOffset(days=7)
    #     start = pd.to_datetime(row.name) - offset
    #     end = pd.to_datetime(row.name) + offset
    #     locs = (
    #             (self.df.index >= start)
    #             & (self.df.index <= end)
    #             & (self.df[f'.{self.ta}_UPPERLIM'] > row[self.ta])
    #             & (self.df[f'.{self.ta}_LOWERLIM'] < row[self.ta])
    #             & (self.df[f'.{self.swin}_UPPERLIM'] > row[self.swin])
    #             & (self.df[f'.{self.swin}_LOWERLIM'] < row[self.swin])
    #             & (self.df[f'.{self.vpd}_UPPERLIM'] > row[self.vpd])
    #             & (self.df[f'.{self.vpd}_LOWERLIM'] < row[self.vpd])
    #     )
    #     return locs, quality

    def _add_newcols(self):
        self._df['.TIMESTAMP'] = self.df.index
        self._df['.PREDICTIONS'] = np.nan
        self._df['.PREDICTIONS_SD'] = np.nan
        self._df['.PREDICTIONS_COUNTS'] = np.nan
        self._df['.PREDICTIONS_QUALITY'] = np.nan
        self._df[f'.{self.swin}_LOWERLIM'] = self.df[self.swin].sub(self.swin_class)
        self._df[f'.{self.swin}_UPPERLIM'] = self.df[self.swin].add(self.swin_class)
        self._df[f'.{self.ta}_LOWERLIM'] = self.df[self.ta].sub(self.ta_class)
        self._df[f'.{self.ta}_UPPERLIM'] = self.df[self.ta].add(self.ta_class)
        self._df[f'.{self.vpd}_LOWERLIM'] = self.df[self.vpd].sub(self.vpd_class)
        self._df[f'.{self.vpd}_UPPERLIM'] = self.df[self.vpd].add(self.vpd_class)


def example():
    from diive.core.io.files import load_parquet

    SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2024_2005-2023\40_FLUXES_L1_IRGA+QCL+LGR_mergeData"
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
