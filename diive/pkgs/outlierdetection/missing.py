from pandas import Series

from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def missing_values(series: Series) -> Series:
    flag_name = f"QCF_MISSING_{series.name}"
    flag = Series(index=series.index, data=False)
    ix_missing = series.loc[series.isnull()].index
    flag.loc[ix_missing] = True
    flag.name = flag_name
    return flag
