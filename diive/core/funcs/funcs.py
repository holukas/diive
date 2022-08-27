import numpy as np


def find_nearest_val(array, value):
    """Find value in array that is nearest to given *value*"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
