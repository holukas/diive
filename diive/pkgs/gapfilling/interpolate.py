import numpy as np
import pandas as pd
from pandas import Series

from diive.pkgs.analyses import gapfinder


def linear_interpolation(series: Series, limit: int = 3) -> Series:
    # Locate gaps
    gapfinder_agg_df = gapfinder.GapFinder(series=series, limit=limit).get_results()

    # Interpolation is done using measured data
    # First, ALL gaps are temporarily filled with interpolated data.
    _series_gf_all = series.interpolate(method='linear', limit=None,
                                        limit_area='inside', limit_direction='both')

    # Second, the info from gapfinder_agg_df is used to keep the desired gap-filled values
    # Iterate over each row in gapfinder_agg_df and check the range of the found gap
    # (a gap can span many records, depending on limit settings).
    # The range info is then used to fill the respective interpolated values from
    # _series_gf_all into focus_df.
    # df[interpolated_values_col] = np.nan
    series_interpolated_vals = pd.Series(index=series.index, data=np.nan)

    for ix, row in gapfinder_agg_df.iterrows():
        gap_start = row['min']
        gap_end = row['max']
        mask = (series.index >= gap_start) & (series.index <= gap_end)
        series_interpolated_vals.loc[mask] = _series_gf_all.loc[mask]

    # Create complete time series, with measured and newly gap-filled values
    series_gf = series.copy().fillna(series_interpolated_vals)
    # df[filled_col].fillna(df[interpolated_values_col], inplace=True)

    return series_gf
