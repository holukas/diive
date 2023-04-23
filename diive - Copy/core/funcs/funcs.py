import numpy as np
from pandas import Series


def zscore(series: Series) -> Series:
    """Calculate the z-score of each record in *series*"""
    mean, std = np.mean(series), np.std(series)
    z_score = np.abs((series - mean) / std)
    return z_score


def find_nearest_val(array, value):
    """Find value in array that is nearest to given *value*"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# CALCULATE WIND DIRECTION AVERAGE (by Kanda)
def winddirection_agg_kanda(angles, agg: str = 'mean') -> float:
    """
    Calculate wind direction aggregation

    Args:
        angles: list of angles to be averaged
        agg: `mean`, `median`, `P25` or `P75`

    Returns:
        aggregated wind direction


    Based on the function by Kanda:
    - https://gist.github.com/jmquintana79/dbca875cbe0c08154c8eca670e74fddf

    """
    ph = angles / 180 * np.pi
    Ds = np.sin(ph)
    Dc = np.cos(ph)

    wd0 = None
    if agg == 'mean':
        wd0 = 180 / np.pi * np.arctan2(Ds.mean(), Dc.mean())
    elif agg == 'median':
        wd0 = 180 / np.pi * np.arctan2(Ds.median(), Dc.median())
    elif agg == 'P25':
        wd0 = 180 / np.pi * np.arctan2(Ds.quantile(.25), Dc.quantile(.25))
    elif agg == 'P75':
        wd0 = 180 / np.pi * np.arctan2(Ds.quantile(.75), Dc.quantile(.75))

    if wd0 < 0:
        wd0 += 360
    mean_wd = wd0
    return mean_wd
