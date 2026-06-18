"""
ANALYSIS: QUANTILES AND PERCENTILES
====================================

Calculate and visualize distribution quantiles and percentiles.
Includes percentile ranges and statistical distribution analysis.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.plotting.scatter import ScatterXY
from diive.core.plotting.styles.format import FormatStyle
from diive.core.utils.console import console as _console


def percentiles101(series: Series, showplot: bool = True, verbose: bool = True) -> DataFrame:
    """Calculate percentiles 0-100 for series.

    Args:
        series: Input series.
        showplot: Shows plot of percentiles 0-100 if *True*.
        verbose: Prints some percentiles if *True*.

    Returns:
        Dataframe with percentiles 0-100 for *series*.

    Example:
        See `examples/analysis/analysis_quantiles.py` for complete examples.
    """
    percentiles_df = pd.DataFrame()
    percentiles_df['PERCENTILE'] = np.arange(0, 101, 1)
    vals_sorted = series.copy().sort_values().dropna()  # Pre-sort array
    for index in percentiles_df.index:
        v = vals_sorted.quantile(percentiles_df.loc[index, 'PERCENTILE'] / 100)
        percentiles_df.loc[index, 'VALUE'] = v

    if verbose:
        show = [0, 1, 5, 16, 25, 50, 75, 84, 95, 99, 100]
        _console.print(f"Showing some percentiles for {series.name}:")
        _console.print(percentiles_df[percentiles_df["PERCENTILE"].isin(show)])

    if showplot:
        scatter = ScatterXY(x=percentiles_df['PERCENTILE'],
                            y=percentiles_df['VALUE'])
        scatter.plot(format_style=FormatStyle(title=f"Percentile values: {series.name}"))

    return percentiles_df
