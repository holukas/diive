from pathlib import Path

import pandas as pd
from pandas import Series


def absolute_limits(series: Series, min: int, max: int, saveplot: str or Path = None) -> Series:
    flag_name = f"QCF_OUTLIER_ABSLIM_{series.name}"
    flag = pd.Series(index=series.index, data=False)
    flag.loc[series < min] = True
    flag.loc[series > max] = True
    flag.name = flag_name

    # print("QA/QC range check")
    # print(f"    Variable: {series.name}")
    # print(f"    Accepted → {range_ok_ix.sum()} values inside range between {min} and {max}")
    # print(f"    Rejected → {flag.loc[series < min].sum()} values below minimum of {min}")
    # print(f"    Rejected → {flag.loc[series > max].sum()} values above maximum of {max}")

    # Plot
    # if saveplot:
    #     from diive.core.plotting.plotfuncs import quickplot_df
    #     quickplot_df([series, series_qc], subplots=False, saveplot=saveplot,
    #                  title=f"Range check: {series.name} must be between {min} and {max}")

    return flag
