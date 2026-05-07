"""Linear interpolation gap-filling examples.

Demonstrates simple linear interpolation for filling small gaps in time series
data, with configurable limits on gap size. Includes two examples showing
conservative (limit=1) vs. generous (limit=5) gap-filling strategies.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import diive as dv


def example_linear_interpolation_limit5():
    """Linear interpolation with limit=5 (fill gaps up to 5 records).

    Demonstrates linear interpolation for gap-filling with a generous limit on
    maximum consecutive missing values to fill (limit=5). Uses HeatmapDateTime
    visualization to compare observed vs. interpolated values side-by-side.

    Shows comprehensive summary statistics before/after gap-filling including:
    - Input data coverage
    - Gap analysis by size
    - Recovery rate (% of data recovered)
    - Gap size distribution (min/median/max)

    Returns:
        None (displays plot and prints summary)
    """
    df = dv.load_exampledata_parquet()
    df = df.loc[df.index.year == 2022].copy()

    series = df['NEE_CUT_REF_orig'].copy()

    # Gap-fill with limit=5 and show detailed summary
    series_gapfilled = dv.linear_interpolation(series=series, limit=5, verbose=True)

    # Plot side-by-side comparison
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    gs = gridspec.GridSpec(1, 2)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_output = fig.add_subplot(gs[0, 1])

    dv.plot_heatmap_datetime(ax=ax_input, series=series).plot()
    dv.plot_heatmap_datetime(ax=ax_output, series=series_gapfilled).plot()

    ax_input.set_title("Observed Data (with gaps)", color='black', fontsize=12, fontweight='bold')
    ax_output.set_title("Gap-Filled (limit=5)", color='black', fontsize=12, fontweight='bold')
    ax_input.tick_params(left=True, right=False, top=False, bottom=True,
                         labelleft=False, labelright=False, labeltop=False, labelbottom=True)
    ax_output.tick_params(left=True, right=False, top=False, bottom=True,
                          labelleft=False, labelright=False, labeltop=False, labelbottom=True)

    fig.show()


def example_linear_interpolation_limit1():
    """Linear interpolation with limit=1 (fill only 1-record gaps).

    Conservative gap-filling that only fills the smallest gaps (limit=1).
    Useful when you want to preserve larger data gaps for other gap-filling
    methods (e.g., Machine Learning) or further analysis.

    Returns:
        None (displays plot and prints summary)
    """
    df = dv.load_exampledata_parquet()
    df = df.loc[df.index.year == 2022].copy()

    series = df['NEE_CUT_REF_orig'].copy()

    # Conservative gap-filling: only fill isolated missing values
    series_gapfilled = dv.linear_interpolation(series=series, limit=1, verbose=True)

    # Plot side-by-side comparison
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    gs = gridspec.GridSpec(1, 2)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_output = fig.add_subplot(gs[0, 1])

    dv.plot_heatmap_datetime(ax=ax_input, series=series).plot()
    dv.plot_heatmap_datetime(ax=ax_output, series=series_gapfilled).plot()

    ax_input.set_title("Observed Data (with gaps)", color='black', fontsize=12, fontweight='bold')
    ax_output.set_title("Gap-Filled (limit=1, conservative)", color='black', fontsize=12, fontweight='bold')
    ax_input.tick_params(left=True, right=False, top=False, bottom=True,
                         labelleft=False, labelright=False, labeltop=False, labelbottom=True)
    ax_output.tick_params(left=True, right=False, top=False, bottom=True,
                          labelleft=False, labelright=False, labeltop=False, labelbottom=True)

    fig.show()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("LINEAR INTERPOLATION GAP-FILLING EXAMPLES")
    print("=" * 80)

    print("\n[Example 1 of 2] Generous limit (fill gaps up to 5 records)")
    print("-" * 80)
    example_linear_interpolation_limit5()

    print("\n[Example 2 of 2] Conservative limit (fill only isolated missing values)")
    print("-" * 80)
    example_linear_interpolation_limit1()
