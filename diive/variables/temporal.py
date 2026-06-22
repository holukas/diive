"""
TEMPORAL: TIME-BASED FEATURES AND CLASSIFICATIONS
==================================================

Create temporal features: day/night classification from solar position,
time-since-condition tracking, and lagged variable copies.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, DatetimeIndex

from diive.variables.radiation import potrad
from diive.core.utils.console import detail


class DaytimeNighttimeFlag:
    """Derive daytime/nighttime flags from potential radiation. See :meth:`__init__`."""

    swinpot_col = 'SW_IN_POT'
    daytime_col = 'DAYTIME'
    nighttime_col = 'NIGHTTIME'

    def __init__(self,
                 timestamp_index: DatetimeIndex,
                 utc_offset: int,
                 lat: float,
                 lon: float,
                 nighttime_threshold: float = 20):
        """Calculate flags to identify daytime and nighttime data from potential radiation.

        Args:
            timestamp_index: Time series index, flags and potential radiation
                are calculated using this index
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
            lat: Latitude
            lon: Longitude
            nighttime_threshold: Threshold for potential radiation below which data
                are flagged as nighttime (W m-2)

        Example:
            See `examples/createvar/daynightflag.py` for complete examples.
        """

        self.timestamp_index = timestamp_index
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.lat = lat
        self.lon = lon

        self.swinpot = None
        self.daytime = None
        self.nighttime = None
        self._df = None

        self._run()

    @property
    def df(self) -> DataFrame:
        """Get dataframe with potential radiation and daytime/nighttime flags"""
        if not isinstance(self._df, DataFrame):
            raise Exception('Data empty.')
        return self._df

    def get_results(self) -> DataFrame:
        """Return dataframe with results"""
        return self.df

    def get_daytime_flag(self) -> Series:
        """Return daytime flag where 1=daytime, 0=nighttime"""
        return self.df[self.daytime_col]

    def get_nighttime_flag(self) -> Series:
        """Return nighttime flag where 0=daytime, 1=nighttime"""
        return self.df[self.nighttime_col]

    def get_swinpot(self) -> Series:
        """Return potential radiation"""
        return self.df[self.swinpot_col]

    def _run(self):
        self._calc_swin_pot()
        self._calc_flags()
        self._assemble()

    def _assemble(self):
        frame = {
            self.swinpot_col: self.swinpot,
            self.daytime_col: self.daytime,
            self.nighttime_col: self.nighttime
        }
        self._df = DataFrame.from_dict(frame)

    def _calc_swin_pot(self):
        """Calculate potential radiation from latitude and longitude"""
        self.swinpot = potrad(timestamp_index=self.timestamp_index,
                              lat=self.lat,
                              lon=self.lon,
                              utc_offset=self.utc_offset)

    def _calc_flags(self):
        self.daytime, self.nighttime = self._daytime_nighttime_flag_from_swinpot()

    def _daytime_nighttime_flag_from_swinpot(self) -> tuple[Series, Series]:
        daytime, nighttime = daytime_nighttime_flag_from_swinpot(
            swinpot=self.swinpot, nighttime_threshold=self.nighttime_threshold)
        return daytime, nighttime


def daytime_nighttime_flag_from_swinpot(swinpot: Series,
                                        nighttime_threshold: float = 20,
                                        daytime_col: str = 'DAYTIME',
                                        nighttime_col: str = 'NIGHTTIME') -> tuple[Series, Series]:
    """
    Create flags to identify daytime and nighttime data

    Args:
        swinpot: Potential short-wave incoming radiation (W m-2)
        nighttime_threshold: Threshold below which data are flagged as nighttime (W m-2)
        daytime_col: Output variable name of the daytime flag
        nighttime_col: Output variable name of the nighttime flag

    Returns:
        Flags as two separate Series:
            *daytime* with flags 1=daytime, 0=not daytime
            *nighttime* with flags 1=nighttime, 0=not nighttime
    """
    daytime = pd.Series(index=swinpot.index, data=np.nan, name=daytime_col)
    daytime.loc[swinpot >= nighttime_threshold] = 1
    daytime.loc[swinpot < nighttime_threshold] = 0
    nighttime = pd.Series(index=swinpot.index, data=np.nan, name=nighttime_col)
    nighttime.loc[swinpot >= nighttime_threshold] = 0
    nighttime.loc[swinpot < nighttime_threshold] = 1
    return daytime, nighttime


class TimeSince:
    """Count consecutive records since last occurrence of a condition.

    Counts the number of records elapsed since a time series last fell outside
    a specified limit range. Useful for tracking dry periods (time since last rain),
    frost periods (time since freezing), warm spells, or any event-based analysis.

    The class maintains a flag indicating whether each record is inside or outside
    the specified range, then counts consecutive "outside range" records backward
    in time to get the time-since count.

    Parameters
    ----------
    series : pd.Series
        Input time series to analyze. Index should be datetime-like for proper
        time-based interpretation.
    upper_lim : float, optional
        Upper limit of the range. Records <= upper_lim (or < upper_lim if
        include_lim=False) are considered "inside range". If None, defaults to
        series maximum (no upper constraint). Default is None.
    lower_lim : float, optional
        Lower limit of the range. Records >= lower_lim (or > lower_lim if
        include_lim=False) are considered "inside range". If None, defaults to
        series minimum (no lower constraint). Default is None.
    include_lim : bool, optional
        If True (default), limits are inclusive (<=, >=).
        If False, limits are exclusive (<, >).
        Example: include_lim=False with lower_lim=0 counts records where value > 0.

    Attributes
    ----------
    series : pd.Series
        Input time series.
    upper_lim : float
        Upper limit (or series max if not specified).
    lower_lim : float
        Lower limit (or series min if not specified).
    include_lim : bool
        Whether limits are inclusive.
    timesince_col : str
        Name of the output column (format: TIMESINCE_{series_name}).
    flag_col : str
        Name of the flag column ("FLAG_IS_OUTSIDE_RANGE").

    Methods
    -------
    calc()
        Calculate time-since counts based on limits. Must be called before
        accessing results.
    get_timesince() -> pd.Series
        Get the time-since values as a Series.
    get_full_results() -> pd.DataFrame
        Get complete results including original series, flags, and time-since counts.

    Examples
    --------
    **Time since last precipitation (dry period detection):**

    >>> df = dv.load_exampledata_parquet()
    >>> prec = df.loc[(df.index.year == 2022) & (df.index.month == 7),
    ...               "PREC_TOT_T1_25+20_1"].copy()
    >>> ts_prec = dv.TimeSince(prec, lower_lim=0, include_lim=False)
    >>> ts_prec.calc()
    >>> max_dry = ts_prec.get_timesince().max()
    >>> print(f"Maximum dry period: {max_dry} records (~{max_dry * 0.5:.1f} hours)")

    **Time since last freezing temperature:**

    >>> temp = df.loc[(df.index.year == 2022) & (df.index.month == 3),
    ...               "Tair_f"].copy()
    >>> ts_temp = dv.TimeSince(temp, upper_lim=0, include_lim=True)
    >>> ts_temp.calc()
    >>> results = ts_temp.get_full_results()
    >>> print(results.head(10))

    See Also
    --------
    examples/variables/feature_timesince.py : Complete usage examples with visualizations.

    Notes
    -----
    - Time-since counts reset whenever a record falls within the range.
    - NaN values are treated as "outside range" to avoid artificial resets.
    - Use include_lim=False for strict inequalities (e.g., precipitation > 0).
    - Use include_lim=True for inclusive boundaries (e.g., temperature <= 0).
    """
    upper_lim_col = "UPPER_LIMIT"
    lower_lim_col = "LOWER_LIMIT"
    flag_col = "FLAG_IS_OUTSIDE_RANGE"

    def __init__(self,
                 series: pd.Series,
                 upper_lim: float = None,
                 lower_lim: float = None,
                 include_lim: bool = True):
        """Initialize TimeSince counter.

        Parameters
        ----------
        series : pd.Series
            Input time series to analyze.
        upper_lim : float, optional
            Upper limit threshold. Default None (no upper constraint).
        lower_lim : float, optional
            Lower limit threshold. Default None (no lower constraint).
        include_lim : bool, optional
            If True, use <= and >= (inclusive). If False, use < and > (exclusive).
            Default is True.
        """
        self.series = series
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim
        self.include_lim = include_lim

        self.timesince_col = f"TIMESINCE_{self.series.name}"
        self._timesince_df = self._setup()

    @property
    def timesince_df(self):
        """Get internal results dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame containing original series, limit columns, flag, and time-since counts.

        Raises
        ------
        Exception
            If data is empty or not initialized.
        """
        if not isinstance(self._timesince_df, pd.DataFrame):
            raise Exception('data is empty')
        return self._timesince_df

    def get_timesince(self) -> pd.Series:
        """Get time-since counts as a Series.

        Must call calc() before accessing results.

        Returns
        -------
        pd.Series
            Integer count of records since last "outside range" occurrence.
            Column name is TIMESINCE_{original_series_name}.
        """
        return self._timesince_df[self.timesince_col].copy()

    def get_full_results(self) -> pd.DataFrame:
        """Get complete results including series, flags, and time-since counts.

        Must call calc() before accessing results.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - {series_name}: Original input series
            - UPPER_LIMIT: Upper limit values
            - LOWER_LIMIT: Lower limit values
            - FLAG_IS_OUTSIDE_RANGE: 0 if inside range, 1 if outside/NaN
            - TIMESINCE_{series_name}: Record count since last inside-range occurrence
        """
        return self.timesince_df.copy()

    def calc(self):
        """Calculate time-since counts.

        Determines which records are inside/outside the specified range, creates
        a binary flag, then counts consecutive records since the last inside-range
        occurrence.

        The algorithm:
        1. Identifies records inside the limit range (flag=0)
        2. Identifies records outside the limit range or with NaN values (flag=1)
        3. Counts consecutive flag=1 records backward in time
        4. Resets count to 0 when flag=0 (inside range) is encountered

        Results are stored internally and accessible via get_timesince() or
        get_full_results().

        Returns
        -------
        None
            Modifies internal state; results accessed via get_timesince() or
            get_full_results().
        """

        if self.include_lim:
            filter_inrange = (
                    (self.timesince_df[self.series.name] <= self.timesince_df[self.upper_lim_col]) &
                    (self.timesince_df[self.series.name] >= self.timesince_df[self.lower_lim_col])
            )
        else:
            filter_inrange = (
                    (self.timesince_df[self.series.name] < self.timesince_df[self.upper_lim_col]) &
                    (self.timesince_df[self.series.name] > self.timesince_df[self.lower_lim_col])
            )

        self._timesince_df.loc[filter_inrange, self.flag_col] = 0
        self._timesince_df.loc[~filter_inrange, self.flag_col] = 1
        self._timesince_df[self.flag_col] = self._timesince_df[self.flag_col].astype(int)

        self._timesince_df.loc[self._timesince_df[self.series.name].isnull(), self.flag_col] = 1

        y = self.timesince_df[self.flag_col].copy()
        yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        self._timesince_df.loc[:, self.timesince_col] = yy.astype(int)

    def _setup(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df[self.series.name] = self.series.copy()

        if self.upper_lim is None:
            df[self.upper_lim_col] = self.series.max()
        else:
            df[self.upper_lim_col] = self.upper_lim

        if self.lower_lim is None:
            df[self.lower_lim_col] = self.series.min()
        else:
            df[self.lower_lim_col] = self.lower_lim

        df[self.flag_col] = pd.NA
        return df


def lagged_variants(df: DataFrame,
                    lag: list[int, int],
                    stepsize: int = 1,
                    exclude_cols: list = None,
                    verbose: int = 0) -> DataFrame:
    """Create lagged variants of variables

    Shifts all records in *df* by *lag* records and stores he lagged variants
    as new columns.

    For example, lagging the two variables `TA` and `SW_IN` with the settings *lag=[-2, 1]*
    and *stepsize=1* creates the new variables `TA-2` (two records before `TA`),
    `TA-1` (one record before `TA`), `TA+1` (one record after `TA`), `SW_IN-2`, `SW_IN-1`,
    and `SW_IN+1`. Note that the minus sign means "before", the plus sign means "after".

    Can be used to investigate correlations between a scalar and the preceding or subsequent
    records of another scalar.

    Note:
        The timestamp index of `df` must be regular and complete, i.e., all timestamps of the
        respective time resolution must be present. Otherwise shifting variables by x records
        might lead to undesirable results.

    Example:
        See `examples/createvar/laggedvariants.py` for complete examples.

    Args:
        df: dataframe that contains variables that will be lagged
        lag: list of integers given as number or records, defining the range of generated lag times
            For example lag=[-3, 2] and stepsize=1 will generated lagged
            variants -3, -2, -1, +1 and +2
        stepsize: stepsize between the different lagged variants given as number of records
            For example lag=[-8, 4] and stepsize=2 will generated lagged
            variants -8, -6, -4, -2 and +2
        exclude_cols: list of column names, these variables will not be lagged
        verbose: if *True*, print more output to console

    Returns:
        input dataframe with added lagged variants
    """

    exclude_cols = [] if not exclude_cols else exclude_cols

    if len(df.columns) == 1:
        if df.columns[0] in exclude_cols:
            raise Exception(f"(!) No lagged variants can be created "
                            f"because there is only one single column in the dataframe "
                            f"({df.columns[0]}) and the same column is also defined in "
                            f"the exclude list (exclude_cols={exclude_cols}). "
                            f"This means there are no data left to lag.")
        return df

    if not isinstance(lag, list):
        raise Exception(f"(!) Error in lag={lag}: No lagged variables can be created "
                        f"because lag is not given as a list, e.g. lag=[-10, -1]. "
                        f"(it was given as lag={lag})")

    if len(lag) != 2:
        raise Exception(f"(!) Error in lag={lag}: No lagged variables can be created "
                        f"because lag must be given as a list with two elements, "
                        f"e.g. lag=[-10, -1]. (it was given as lag={lag})")

    for _lag in lag:
        if not isinstance(_lag, int):
            raise TypeError(f"(!) Error in lag={lag}: No lagged variables can be created "
                            f"because {_lag} is not an integer.")

    _included = []
    _excluded = []

    lagsteps = range(lag[0], lag[1] + 1, stepsize)

    for col in df.columns:
        if isinstance(exclude_cols, list):

            if col in exclude_cols:
                _excluded.append(col)
                continue

            for lagstep in lagsteps:
                if lagstep < 0:
                    stepname = f".{col}{lagstep}"
                    _shift = abs(lagstep)
                elif lagstep > 0:
                    stepname = f".{col}+{lagstep}"
                    _shift = -lagstep
                else:
                    continue

                n_missing_vals_before = int(df[col].isnull().sum())
                df[stepname] = df[col].shift(_shift)
                n_missing_vals_after = int(df[stepname].isnull().sum())
                if n_missing_vals_before == 0 and n_missing_vals_after > 0:
                    df[stepname] = df[stepname].bfill(limit=n_missing_vals_after)
                    df[stepname] = df[stepname].ffill(limit=n_missing_vals_after)
            _included.append(col)

    if verbose:
        detail(f"Added lagged variants for: {_included} (lags between {lag[0]} and {lag[1]} "
               f"with stepsize {stepsize}), no lagged variants for: {_excluded}. "
               f"Shifting the time series created gaps which were then filled with the nearest value.",
               verbose=verbose)
    return df
