import numpy as np
import pandas as pd
from pandas import Series
from pathlib import Path

def setto_threshold(series: Series, threshold: float, type: str, show:bool=False,
                    saveplot:str or Path=None) -> tuple[Series, Series]:
    """Set values above or below a threshold value to threshold value"""

    print(f"Set {series.name} to {type} threshold {threshold}  ...")

    outname = series.name
    series.name = "input_data"

    # Create empty flag
    flag = pd.Series(index=series.index, data=np.nan)

    # Detect values over threshold
    over_thres_ix = range_ok_ix = None
    if type == 'max':
        over_thres_ix = series > threshold
        range_ok_ix = series <= threshold
    if type == 'min':
        over_thres_ix = series < threshold
        range_ok_ix = series >= threshold

    flag.loc[over_thres_ix] = 1
    flag.loc[range_ok_ix] = 0

    print("QA/QC set to threshold value")
    print(f"    Variable: {series.name}")
    if type == 'max':
        print(f"    Accepted → {range_ok_ix.sum()} values below max threshold of {threshold}")
        print(
            f"    Corrected → {over_thres_ix.sum()} values above max threshold of {threshold} were set to {threshold}")
    if type == 'min':
        print(f"    Accepted → {range_ok_ix.sum()} values above min threshold of {threshold}")
        print(
            f"    Corrected → {over_thres_ix.sum()} values below min threshold of {threshold} were set to {threshold}")

    corrected_ix = flag == 1
    series_qc = series.copy()
    series_qc.loc[corrected_ix] = threshold
    series_qc.rename(outname, inplace=True)

    # Plot
    if saveplot:
        from diive.core.plotting.plotfuncs import quickplot_df
        quickplot_df([series, series_qc], subplots=False,
                     saveplot=saveplot, title=f"Set {series.name} to {type} threshold {threshold}")

    return series_qc, flag


