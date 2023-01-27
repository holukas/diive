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
