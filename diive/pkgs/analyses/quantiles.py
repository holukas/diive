import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.plotting.scatter import Scatter


def percentiles(series: Series, showplot: bool = True) -> DataFrame:
    percentiles_df = pd.DataFrame()
    percentiles_df['PERCENTILE'] = np.arange(0, 101, 1)
    vals_sorted = series.copy().sort_values().dropna()  # Pre-sort array
    for index, row in percentiles_df.iterrows():
        v = vals_sorted.quantile(row['PERCENTILE'] / 100)
        percentiles_df.loc[index, 'VALUE'] = v

    show = [0, 1, 5, 16, 25, 50, 75, 84, 95, 99, 100]
    print(f"Showing some percentiles for {series.name}:")
    print(percentiles_df[percentiles_df["PERCENTILE"].isin(show)])

    if showplot:
        scatter = Scatter(x=percentiles_df['PERCENTILE'],
                          y=percentiles_df['VALUE'],
                          title=f"Percentile values: {series.name}")
        scatter.plot(nbins=10)

    # plt.plot(percentiles_df['PERCENTILE'], percentiles_df['VALUE'],
    #          color='black', alpha=0.9, ls='-', lw=theme.WIDTH_LINE_DEFAULT,
    #          marker='o', markeredgecolor='none', ms=theme.SIZE_MARKER_DEFAULT, zorder=99, label='value')
    # plt.show()

    return percentiles_df
