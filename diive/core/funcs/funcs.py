import numpy as np
from pandas import Series


def validate_id_string(idstr: str):
    if idstr:
        # idstr = idstr if idstr.endswith('_') else f'{idstr}_'
        idstr = idstr if idstr.startswith('_') else f'_{idstr}'
    return idstr


def filter_strings_by_elements(list1: list[str], list2: list[str]) -> list[str]:
    """Returns a list of strings from list1 that contain all of the elements in list2.

    The function uses a set to keep track of the elements in list2, which makes it more
    efficient than iterating over the list twice with this one-liner:
        result = [s1 for s1 in list1 if all(s2 in str(s1) for s2 in list2)]

    Args:
        list1: A list of strings.
        list2: A list of elements to check for in each string in list1.

    Returns:
        A list of strings from list1 that contain all of the elements in list2.
    """
    if not list1 or not list2:
        return []

    elements_in_other_list = set(list2)
    result = []
    for s1 in list1:
        if all(s2 in str(s1) for s2 in elements_in_other_list):
            result.append(s1)
    return result


def zscore(series: Series, absolute: bool = True) -> Series:
    """Calculate the z-score (absolute) of each record in *series*"""
    mean, std = np.mean(series), np.std(series)
    if absolute:
        z_score = np.abs((series - mean) / std)
    else:
        z_score = (series - mean) / std
    return z_score


def val_from_zscore(series: Series, zscore: float, absolute: bool = True) -> float:
    """Calculate the value for a specific z-score"""
    mean, std = np.mean(series), np.std(series)
    value = (zscore * std) + mean  # from: z_score = (series - mean) / std
    return value


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
