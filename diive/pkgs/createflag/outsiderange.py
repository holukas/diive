from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Series


def range_check(series: Series, min: int, max: int, show: bool = False, saveplot: str or Path = None) -> tuple[
    Series, Series]:
    series_qc = series.copy()
    flag = pd.Series(index=series.index, data=np.nan)

    below_min_ix = series < min
    above_max_ix = series > max
    range_ok_ix = (series >= min) & (series <= max)

    flag.loc[below_min_ix] = 1
    flag.loc[above_max_ix] = 1
    flag.loc[range_ok_ix] = 0

    print("QA/QC range check")
    print(f"    Variable: {series.name}")
    print(f"    Accepted → {range_ok_ix.sum()} values inside range between {min} and {max}")
    print(f"    Rejected → {below_min_ix.sum()} values below minimum of {min}")
    print(f"    Rejected → {above_max_ix.sum()} values above maximum of {max}")

    reject_ix = flag == 1
    series_qc.loc[reject_ix] = np.nan

    # Plot
    if saveplot:
        from diive.common.plotting.plotfuncs import quickplot_df
        quickplot_df([series, series_qc], subplots=False, saveplot=saveplot,
                     title=f"Range check: {series.name} must be between {min} and {max}")

    return series_qc, flag
