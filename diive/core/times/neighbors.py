from pandas import DataFrame

from diive.core.funcs.funcs import find_nearest_val


def neighboring_years(df: DataFrame, verbose: bool = False) -> dict:
    """Collect data for year and its two neighboring years"""

    uniq_years = list(df.index.year.unique())
    yearpools_dict = {}

    # For each year, build model from the 2 neighboring years

    for ix, year in enumerate(uniq_years):
        yearpools_dict[str(year)] = {}  # Init dict for this year
        poolyears = []
        _uniq_years = uniq_years.copy()
        _uniq_years.remove(year)
        poolyears.append(year)

        if _uniq_years:
            nearest_1 = find_nearest_val(array=_uniq_years, value=year)
            _uniq_years.remove(nearest_1)
            poolyears.append(nearest_1)

            if _uniq_years:
                nearest_2 = find_nearest_val(array=_uniq_years, value=year)
                poolyears.append(nearest_2)

        poolyears = sorted(poolyears)

        yearpools_dict[str(year)]['poolyears'] = poolyears

        if verbose:
            print(f"[{neighboring_years.__name__}] Assigned {poolyears} to data pool for {year}.")

        yearpools_dict[str(year)]['df'] = _limit_yearpool_data(df=df, poolyears=poolyears)

    return yearpools_dict


def _limit_yearpool_data(df: DataFrame, poolyears: list) -> DataFrame:
    """Get data for poolyears"""
    firstyear = min(poolyears)
    lastyear = max(poolyears)
    yearpool_df = df.loc[(df.index.year >= firstyear) & (df.index.year <= lastyear), :].copy()
    return yearpool_df
