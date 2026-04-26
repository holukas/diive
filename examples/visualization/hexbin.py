"""
Examples for HexbinPlot visualization.

HexbinPlot aggregates flux values into 2D hexagonal bins of driver variables
(e.g., soil temperature vs water-filled pore space).

Run this script to display example plots:
    python examples/visualization/hexbin.py
"""

import numpy as np

import diive as dv


def example_hexbin_percentile_normalized():
    """HexbinPlot with percentile normalization of driver axes.

    Converts x and y axes to 0-100 percentile scale for normalized comparison
    across different variable ranges.
    """
    df = dv.load_exampledata_parquet()

    # Select growing season only
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()
    df = df[['Tair_f', 'VPD_f', 'NEE_CUT_REF_f']].dropna()

    hm = dv.hexbin(
        x=df['Tair_f'],
        y=df['VPD_f'],
        z=df['NEE_CUT_REF_f'],
        normalize_axes=True,
        gridsize=11,
        xlabel='Air temperature (percentile)',
        ylabel='Vapor pressure deficit (percentile)',
        zlabel='Net ecosystem exchange',
        figsize=(8, 6),
        cmap='RdYlBu_r'
    )
    hm.show()


def example_hexbin_absolute_values_with_mean():
    """HexbinPlot with absolute values and mean aggregation.

    Uses original variable values on axes and aggregates z-values using
    mean (instead of default median) within each hexagonal bin.
    """
    df = dv.load_exampledata_parquet()

    # Select growing season only
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()
    df = df[['Tair_f', 'VPD_f', 'NEE_CUT_REF_f']].dropna()

    hm = dv.hexbin(
        x=df['Tair_f'],
        y=df['VPD_f'],
        z=df['NEE_CUT_REF_f'],
        normalize_axes=False,
        gridsize=15,
        reduce_C_function=np.mean,
        xlabel='Air temperature (°C)',
        ylabel='Vapor pressure deficit (hPa)',
        zlabel='Mean NEE (µmol m⁻² s⁻¹)',
        figsize=(8, 6),
        cb_digits_after_comma=0,
        cmap='RdYlBu_r'
    )
    hm.show()


def example_hexbin_with_value_overlay():
    """HexbinPlot with aggregated values displayed on hexagons.

    Overlays the aggregated z-values directly on each hexagon center
    for easy reading of specific bins.
    """
    df = dv.load_exampledata_parquet()

    # Select growing season only
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()
    df = df[['Tair_f', 'VPD_f', 'NEE_CUT_REF_f']].dropna()

    hm = dv.hexbin(
        x=df['Tair_f'],
        y=df['VPD_f'],
        z=df['NEE_CUT_REF_f'],
        normalize_axes=True,
        gridsize=20,
        reduce_C_function=np.mean,
        xlabel='Air temperature (percentile)',
        ylabel='Vapor pressure deficit (percentile)',
        zlabel='Mean NEE (µmol m⁻² s⁻¹)',
        figsize=(10, 8),
        mincnt=5,
        cb_digits_after_comma=0,
        show_values=True,
        show_values_fontsize=8,
        show_values_n_dec_places=0,
        show_values_color='black',
        cmap='RdYlBu_r'
    )
    hm.show()


if __name__ == '__main__':
    print("Running HexbinPlot examples...")

    print("\n1. HexbinPlot with percentile normalization...")
    example_hexbin_percentile_normalized()

    print("\n2. HexbinPlot with absolute values and mean aggregation...")
    example_hexbin_absolute_values_with_mean()

    print("\n3. HexbinPlot with value overlay on hexagons...")
    example_hexbin_with_value_overlay()

    print("\nAll examples completed!")
