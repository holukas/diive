from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def zscoreiqr(series: Series, factor: float = 4, showplot: bool or Path = True) -> Series:
    """
    Remove outliers based on the z-score of interquartile range data

    Data are divided into 8 groups based on quantiles. The z-score is calculated
    for each data points in the respective group and based on the mean and SD of
    the respective group. The z-score threshold to identify outlier data is
    calculated as the max of z-scores found in IQR data multiplied by *factor*.
    z-scores above the threshold are marked as outliers.

    Args:
        series:
        factor:
        showplot:

    Returns:
        Series with flag where 1 = outlier, 0 = no outlier

    kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/

    """
    flag_name = f"QCF_OUTLIER_ZSCOREIQR_{series.name}"
    flag = pd.Series(index=series.index, data=False)

    # prev_diff = np.abs(series - series.shift(1))
    # next_diff = np.abs(series - series.shift(-1))
    # s = prev_diff + next_diff
    # s = s.rolling(9, center=True).mean()
    # s = series.copy()
    # good_coll = pd.Series(index=s.index, data=False)
    # bad_coll = pd.Series(index=s.index, data=False)
    # n_outliers_prev = None

    # Collect flags for good and bad data
    good_coll = pd.Series(index=series.index, data=False)
    bad_coll = pd.Series(index=series.index, data=False)

    # Working data
    s = series.copy()

    # prev_diff = np.abs(series - series.shift(1))
    # next_diff = np.abs(series - series.shift(-1))
    # s = prev_diff + next_diff
    # s = s**2

    threshold = 10
    mean, std = np.mean(s), np.std(s)
    z_score = np.abs((s - mean) / std)
    # plt.scatter(z_score.index, z_score)
    # plt.show()
    good = z_score < threshold
    bad = z_score > threshold
    good = good[good]
    bad = bad[bad]
    good_coll.loc[good.index] = True
    bad_coll.loc[bad.index] = True

    n_outliers_prev = 0
    outliers = True
    iter = 0
    while outliers:
        iter += 1
        print(f"Starting iteration#{iter} ... ")

        # group, bins = pd.cut(s, bins=2, retbins=True, duplicates='drop')
        group, bins = pd.qcut(s, q=8, retbins=True, duplicates='drop')
        df = pd.DataFrame(s)
        df['_GROUP'] = group
        grouped = df.groupby('_GROUP')
        for ix, group_df in grouped:
            # vardata = group_df[varname].rolling(window=3, center=True).median()
            # mean = group_df[varname].rolling(window=1000, center=True).mean()
            # sd = group_df[varname].rolling(window=1000, center=True).std()
            vardata = group_df[s.name]
            mean = np.mean(vardata)
            sd = np.std(vardata)
            z_score = np.abs((vardata - mean) / sd)
            # plt.scatter(z_score.index, z_score)
            # plt.show()
            threshold = _detect_z_threshold_from_iqr(series=vardata, factor=factor, quantiles=8)
            # threshold = 10
            good = z_score < threshold
            bad = z_score > threshold
            good = good[good]
            bad = bad[bad]
            good_coll.loc[good.index] = True
            bad_coll.loc[bad.index] = True
        n_outliers = bad_coll.sum()
        new_n_outliers = n_outliers - n_outliers_prev
        print(f"Found {new_n_outliers} outliers during iteration#{iter} ... ")
        if new_n_outliers > 0:
            n_outliers_prev = n_outliers
            s.loc[bad_coll] = np.nan
            # outliers = False  # Set to *False* means outlier removal runs one time only
            outliers = True  # *True* means run outlier removal several times until all outliers removed
        else:
            print(f"No more outliers found during iteration#{iter}, outlier search finished.")
            outliers = False

    flag.loc[bad_coll] = True
    flag.loc[good_coll] = False
    flag.name = flag_name

    threshold = 10
    mean, std = np.mean(s), np.std(s)
    z_score = np.abs((s - mean) / std)
    good = z_score < threshold
    bad = z_score > threshold
    good = good[good]
    bad = bad[bad]
    good_coll.loc[good.index] = True
    bad_coll.loc[bad.index] = True

    # plt.scatter(z_score.loc[good.index].index, z_score.loc[good.index])
    # plt.scatter(s.loc[good.index].index, s.loc[good.index])
    # plt.show()

    print(f"Total found outliers: {bad_coll.sum()} values")
    # print(f"z-score of {threshold} corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")

    # print("QA/QC range check")
    # print(f"    Variable: {series.name}")
    # print(f"    Accepted → {range_ok_ix.sum()} values inside range between {min} and {max}")
    # print(f"    Rejected → {flag.loc[series < min].sum()} values below minimum of {min}")
    # print(f"    Rejected → {flag.loc[series > max].sum()} values above maximum of {max}")

    # Plot
    if showplot:
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        visual_scatter = np.random.normal(size=series.size)
        ax.scatter(series[good_coll], visual_scatter[good_coll], s=2, label="Good", color="#4CAF50")
        ax.scatter(series[bad_coll], visual_scatter[bad_coll], s=8, label="Bad", color="#F44336")
        ax.legend()
        fig.show()

        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot_date(series[good_coll].index, series[good_coll], label="Good", color="#4CAF50")
        ax.plot_date(series[bad_coll].index, series[bad_coll], marker="x", label="Bad", color="#F44336")
        ax.legend()
        fig.show()

    return flag


def _detect_z_threshold_from_iqr(series: Series, factor:float=5, quantiles:int=8) -> float:
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
    print(f"Detected threshold for z-value from IQR data: {threshold} "
          f"(max z-value in IQR data multiplied by factor {factor})")
    return threshold
