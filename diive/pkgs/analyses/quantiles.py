import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.plotting.scatter import ScatterXY


def percentiles(series: Series, showplot: bool = True, verbose: bool = True) -> DataFrame:
    """Show percentiles 0-100 for series.

    Args:
        series: Input series.
        showplot: Shows plot of percentiles 0-100 if *True*.
        verbose: Prints some percentiles if *True*.

    Returns:
        Dataframe with percentiles 0-100 for *series*.

    - Example notebook available in:
        notebooks/Analyses/Percentiles.ipynb
    """
    percentiles_df = pd.DataFrame()
    percentiles_df['PERCENTILE'] = np.arange(0, 101, 1)
    vals_sorted = series.copy().sort_values().dropna()  # Pre-sort array
    for index, row in percentiles_df.iterrows():
        v = vals_sorted.quantile(row['PERCENTILE'] / 100)
        percentiles_df.loc[index, 'VALUE'] = v

    if verbose:
        show = [0, 1, 5, 16, 25, 50, 75, 84, 95, 99, 100]
        print(f"Showing some percentiles for {series.name}:")
        print(percentiles_df[percentiles_df["PERCENTILE"].isin(show)])

    if showplot:
        scatter = ScatterXY(x=percentiles_df['PERCENTILE'],
                            y=percentiles_df['VALUE'],
                            title=f"Percentile values: {series.name}")
        scatter.plot()

    # plt.plot(percentiles_df['PERCENTILE'], percentiles_df['VALUE'],
    #          color='black', alpha=0.9, ls='-', lw=theme.WIDTH_LINE_DEFAULT,
    #          marker='o', markeredgecolor='none', ms=theme.SIZE_MARKER_DEFAULT, zorder=99, label='value')
    # plt.show()

    return percentiles_df


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    percentiles_df = percentiles(series=df['Tair_f'])
    print(percentiles_df)



if __name__ == "__main__":
    example()
