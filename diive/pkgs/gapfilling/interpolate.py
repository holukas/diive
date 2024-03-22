import numpy as np
import pandas as pd
from pandas import Series

from diive.pkgs.analyses import gapfinder


def linear_interpolation(series: Series, limit: int = 3) -> Series:
    """Fill gaps in series with a linear interpolation up to a specified limit."""
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
        gap_start = row['GAP_START']
        gap_end = row['GAP_END']
        mask = (series.index >= gap_start) & (series.index <= gap_end)
        series_interpolated_vals.loc[mask] = _series_gf_all.loc[mask]

    # Create complete time series, with measured and newly gap-filled values
    series_gf = series.copy().fillna(series_interpolated_vals)
    # df[filled_col].fillna(df[interpolated_values_col], inplace=True)

    return series_gf


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    df = df.loc[df.index.year == 2022].copy()

    series = df['NEE_CUT_REF_orig'].copy()

    series_gapfilled = linear_interpolation(series=series, limit=5)

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    gs = gridspec.GridSpec(1, 2)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_output = fig.add_subplot(gs[0, 1])
    HeatmapDateTime(ax=ax_input, series=series).plot()
    HeatmapDateTime(ax=ax_output, series=series_gapfilled).plot()
    ax_input.set_title("Observed", color='black')
    ax_output.set_title("Interpolated", color='black')
    ax_input.tick_params(left=True, right=False, top=False, bottom=True,
                         labelleft=False, labelright=False, labeltop=False, labelbottom=True)
    ax_output.tick_params(left=True, right=False, top=False, bottom=True,
                          labelleft=False, labelright=False, labeltop=False, labelbottom=True)
    fig.show()


if __name__ == '__main__':
    example()
