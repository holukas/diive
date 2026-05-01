import pandas as pd


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
    examples/createvar/timesince.py : Complete usage examples with visualizations.

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

        # Get locations where series is within the specified limits
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

        self._timesince_df.loc[filter_inrange, self.flag_col] = 0  # Inside range
        self._timesince_df.loc[~filter_inrange, self.flag_col] = 1  # Outside range, note: this also counts NaNs as 1
        self._timesince_df[self.flag_col] = self._timesince_df[self.flag_col].astype(int)

        # print(self.timesince_df[self.timesince_df[self.flag_col] == 1].describe())

        # Set all NaN values to 1
        # OLD: Set all NaN values to 0, necessary for correct summations of values outside range
        # OLD: Otherwise, time periods with gaps would also be counted as "outside range", i.e. 1.
        self._timesince_df.loc[self._timesince_df[self.series.name].isnull(), self.flag_col] = 1

        # fantastic: https://stackoverflow.com/questions/27626542/counting-consecutive-positive-value-in-python-array
        y = self.timesince_df[self.flag_col].copy()
        yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        self._timesince_df.loc[:, self.timesince_col] = yy.astype(int)

    def _setup(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df[self.series.name] = self.series.copy()

        # Upper limit
        if self.upper_lim is None:
            df[self.upper_lim_col] = self.series.max()
        else:
            df[self.upper_lim_col] = self.upper_lim

        # Lower limit
        if self.lower_lim is None:
            df[self.lower_lim_col] = self.series.min()
        else:
            df[self.lower_lim_col] = self.lower_lim

        df[self.flag_col] = pd.NA
        return df
