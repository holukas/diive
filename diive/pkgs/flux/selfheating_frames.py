import numpy as np
import pandas as pd


def detect_daytime(swin):
    """Detect daytime and nighttime periods from global radiation

    Generates column that contains daytime flag (1=daytime, 0=nighttime).

    :param swin: series
    :return:
    series: series indicating daytime and nighttime
    daytime_filter: indices of daytime rows
    nighttime_filter: indices of nighttime rows
    """
    nighttime_filter = swin <= 20
    daytime_filter = swin > 20
    daytime_series = pd.Series(index=swin.index)
    daytime_series.loc[nighttime_filter] = 0
    daytime_series.loc[daytime_filter] = 1
    return daytime_series, daytime_filter, nighttime_filter


def init_scaling_factors_df(num_classes):
    """Initialize df that collects results for scaling factors
    - Needs to be initialized with a Multiindex
    - Multiindex consists of two indices: (1) daytime and (2) sonic temperature class
    """
    _list_class_var_classes = [*range(0, num_classes)]
    _iterables = [[1, 0], _list_class_var_classes]
    _multi_ix = pd.MultiIndex.from_product(_iterables, names=["daytime_ix", "sonic_temperature_class_ix"])
    scaling_factors_df = pd.DataFrame(index=_multi_ix)

    cols = ['SF_MEDIAN', 'SOS_MEDIAN', 'NUMVALS_AVG', 'SF_Q25',
            'SF_Q75', 'SF_Q01', 'SF_Q99']

    for col in cols:
        scaling_factors_df[col] = np.nan

    return scaling_factors_df
