import numpy as np
import pandas as pd




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
