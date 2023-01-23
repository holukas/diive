import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class zScoreIQR(FlagBase):
    """
    Identify outliers based on the z-score of interquartile range data

    Data are divided into 8 groups based on quantiles. The z-score is calculated
    for each data points in the respective group and based on the mean and SD of
    the respective group. The z-score threshold to identify outlier data is
    calculated as the max of z-scores found in IQR data multiplied by *factor*.
    z-scores above the threshold are marked as outliers.
    ...

    Methods:
        calc(factor: float = 4): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/

    """
    flagid = 'OUTLIER_ZSCOREIQR'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.verbose = False

    def calc(self, factor: float = 4, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        self.showplot = showplot
        self.verbose = verbose
        self.reset()
        ok, rejected = self._flagtests(factor=factor)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self, factor: float = 4) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Collect flags for good and bad data
        ok_coll = pd.Series(index=self.series.index, data=False)
        rejected_coll = pd.Series(index=self.series.index, data=False)

        # Working data
        s = self.series.copy()
        s = s.dropna()

        # First run with high threshold, weak detection
        threshold = 10
        mean, std = np.mean(s), np.std(s)
        z_score = np.abs((s - mean) / std)
        # plt.scatter(z_score.index, z_score)
        # plt.show()
        ok = z_score <= threshold
        ok = ok[ok]
        ok_coll.loc[ok.index] = True
        rejected = z_score > threshold
        rejected = rejected[rejected]
        rejected_coll.loc[rejected.index] = True

        n_outliers_prev = 0
        outliers = True
        iter = 0
        while outliers:
            iter += 1
            if self.verbose: print(f"Starting iteration#{iter} ... ")

            # group, bins = pd.cut(s, bins=2, retbins=True, duplicates='drop')
            group, bins = pd.qcut(s, q=8, retbins=True, duplicates='drop')
            df = pd.DataFrame(s)
            df['_GROUP'] = group
            grouped = df.groupby('_GROUP')
            for ix, group_df in grouped:
                vardata = group_df[s.name]
                mean = np.mean(vardata)
                sd = np.std(vardata)
                z_score = np.abs((vardata - mean) / sd)
                # plt.scatter(z_score.index, z_score)
                # plt.show()
                threshold = self._detect_z_threshold_from_iqr(series=vardata, factor=factor, quantiles=8)
                ok = z_score < threshold
                ok = ok[ok]
                ok_coll.loc[ok.index] = True
                rejected = z_score > threshold
                rejected = rejected[rejected]
                rejected_coll.loc[rejected.index] = True
            n_outliers = rejected_coll.sum()
            new_n_outliers = n_outliers - n_outliers_prev
            if self.verbose: print(f"Found {new_n_outliers} outliers during iteration#{iter} ... ")
            if new_n_outliers > 0:
                n_outliers_prev = n_outliers
                s.loc[rejected_coll] = np.nan
                # outliers = False  # Set to *False* means outlier removal runs one time only
                outliers = True  # *True* means run outlier removal several times until all outliers removed
            else:
                if self.verbose: print(f"No more outliers found during iteration#{iter}, outlier search finished.")
                outliers = False

        # Convert to index
        ok_coll = ok_coll[ok_coll].index
        rejected_coll = rejected_coll[rejected_coll].index

        if self.verbose: print(f"Total found outliers: {len(rejected_coll)} values")
        # print(f"z-score of {threshold} corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")

        if self.showplot: self._plot(ok_coll, rejected_coll)

        return ok_coll, rejected_coll

    def _detect_z_threshold_from_iqr(self, series: Series, factor: float = 5, quantiles: int = 8) -> float:
        # First detect the threshold for the z-value
        # - Datapoints where z-value > threshold will be marked as outliers
        # - The threshold is detected from z-values found for the interquartile range
        #   of the data
        # Divide data into 8 quantile groups
        # - This means that we then have groups 0-7
        # - Groups 2-5 correspond to the IQR
        #         |- IQR-|
        # - (0 1) 2 3 4 5 (6 7)
        group, bins = pd.qcut(series, q=quantiles, retbins=True, duplicates='drop')
        df = pd.DataFrame(series)
        df['_GROUP'] = group
        grouped = df.groupby('_GROUP')
        _counter = -1
        zvals_iqr = []
        for ix, group_df in grouped:
            _counter += 1
            if (_counter >= 2) & (_counter <= 5):
                vardata = group_df[series.name]
                mean = np.mean(vardata)
                sd = np.std(vardata)
                z_score = np.abs((vardata - mean) / sd)
                zvals_iqr.append(z_score.max())
        threshold = max(zvals_iqr) * factor
        if self.verbose: print(f"Detected threshold for z-value from IQR data: {threshold} "
                               f"(max z-value in IQR data multiplied by factor {factor})")
        return threshold

    def _plot(self, ok_coll, rejected_coll):
        # Plot
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot_date(self.series[ok_coll].index, self.series[ok_coll],
                     label="OK", color="#4CAF50")
        ax.plot_date(self.series[rejected_coll].index, self.series[rejected_coll],
                     label="outlier (rejected)", color="#F44336", marker="X")
        ax.legend()
        fig.show()
