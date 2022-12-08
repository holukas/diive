from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def localsd(series: Series, n_sd: int = 7, showplot: bool or Path = True) -> Series:
    """

    Args:
        series:
        n_sd:
        showplot:

    Returns:


    """
    flag_name = f"QCF_OUTLIER_LOCALSD_{series.name}"
    flag = pd.Series(index=series.index, data=False)
    winsize = int(len(series) / 20)
    mean = series.rolling(window=winsize, center=True, min_periods=3).median()
    sd = series.rolling(window=winsize, center=True, min_periods=3).std()
    upper_limit = mean + sd.multiply(n_sd)
    lower_limit = mean - sd.multiply(n_sd)

    _d = pd.concat([series, mean, upper_limit, lower_limit], axis=1)
    _d.plot()
    plt.show()

    good = (series < upper_limit) & (series > lower_limit)
    bad = (series > upper_limit) | (series < lower_limit)
    flag.loc[bad] = True
    flag.loc[good] = False
    flag.name = flag_name

    print(f"Rejection {flag.sum()} points")

    # Plot
    if showplot:
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot_date(series[good].index, series[good], label="Good", color="#4CAF50")
        ax.plot_date(series[bad].index, series[bad], marker="x", label="Bad", color="#F44336")
        ax.plot_date(upper_limit.index, upper_limit, marker="x", label="Bad", color="#F44336")
        ax.plot_date(lower_limit.index, lower_limit, marker="x", label="Bad", color="#F44336")
        ax.legend()
        fig.show()

    return flag
