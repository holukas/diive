import datetime as dt
import fnmatch
import time
from typing import Literal, Union, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex, Index
from pandas.tseries.frequencies import to_offset

from diive.core.utils.console import rule, info, warn, detail

# Define the default mapping outside the function for clean reference and mutability safety
DEFAULT_SEASON_MAP = {
    # Season ID: [List of months (int)]
    1: [3, 4, 5],  # Spring
    2: [6, 7, 8],  # Summer
    3: [9, 10, 11],  # Autumn
    4: [12, 1, 2]  # Winter
}


class TimestampSanitizer:
    """
    Validate and prepare timestamps for time series data processing.

    Cleans and validates datetime indices in 10 steps: naming convention, format
    conversion, NaT/duplicate removal, sorting, monotonicity validation, frequency
    detection, gap filling (optional), and middle-of-period conversion (optional).
    Each step can be enabled or disabled independently.

    The processing pipeline (in order):
    1. Validate timestamp naming convention
    2. Convert timestamp index to datetime format
    3. Remove rows with missing timestamps (NaT)
    4. Sort timestamp in ascending order
    5. Remove duplicate timestamps (keep last)
    6. Validate timestamp monotonicity (if sorting enabled)
    7. Detect time resolution from timestamp
    8. Validate detected frequency against expected frequency (if provided)
    9. Make timestamp continuous without date gaps
    10. Convert to middle-of-period timestamp (optional)

    Attributes
    ----------
    data : Union[Series, DataFrame]
        The processed data with validated timestamp index.
    inferred_freq : str
        Detected time resolution of the timestamp index.

    Raises
    ------
    TypeError
        If data is None or not a Series/DataFrame.
    ValueError
        If data is empty, has no valid index, or timestamp naming is invalid.
    ValueError
        If nominal_freq does not match detected frequency.
    RuntimeError
        If any processing step fails unexpectedly.

    Warnings
    --------
    - **Regularization side effect**: When regularize=True, gaps are filled with
      NaN data rows. This may increase the number of rows significantly.
    - **No rollback**: If processing fails mid-pipeline, data is partially
      transformed and returned as-is (not rolled back to original state).
    - **Frequency detection**: Detection may fail if timestamps are irregular.
      Use nominal_freq=None to skip frequency validation in such cases.

    See Also
    --------
    examples/timeseries/timestamp_sanitizer.py : Examples with clean data,
        minor issues, and badly broken timestamps.
    """

    def __init__(self,
                 data: Union[Series, DataFrame],
                 output_middle_timestamp: bool = True,
                 validate_naming: bool = True,
                 convert_to_datetime: bool = True,
                 remove_index_nat: bool = True,
                 sort_ascending: bool = True,
                 remove_duplicates: bool = True,
                 regularize: bool = True,
                 nominal_freq: str = None,
                 verbose: bool = False):
        """
        Initialize TimestampSanitizer and process timestamp index.

        Parameters
        ----------
        data : Union[Series, DataFrame]
            Data with timestamp index to be validated and processed.
        output_middle_timestamp : bool, optional
            Convert timestamp index to show middle of averaging period. Default is True.
        validate_naming : bool, optional
            Check if timestamp is correctly named. Allowed names are 'TIMESTAMP_END',
            'TIMESTAMP_START', and 'TIMESTAMP_MIDDLE'. Default is True.
        convert_to_datetime : bool, optional
            Convert timestamp index to datetime format. Default is True.
        remove_index_nat : bool, optional
            Remove rows without timestamp (NaT values). Default is True.
        sort_ascending : bool, optional
            Sort timestamp in ascending order. Default is True.
        remove_duplicates : bool, optional
            Remove duplicate timestamps (keep last occurrence). Default is True.
        regularize : bool, optional
            Generate continuous timestamp without date gaps. Default is True.
        nominal_freq : str, optional
            Expected time resolution of data timestamp index. If provided, detected
            frequency is validated against this. Raises ValueError if they don't match.
            Examples: '10s', '5s', 's', '30min', '5min', 'min', '1h', '3h', 'h'.
            Default is None (no frequency validation).
        verbose : bool, optional
            Print status messages during processing. Default is False.

        Examples
        --------
        **Basic usage with default settings:**

        >>> import pandas as pd
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet()
        >>> series = df['NEE_CUT_REF_f'].copy()
        >>> sanitizer = dv.TimestampSanitizer(data=series, verbose=False)
        >>> clean_series = sanitizer.get()

        **With frequency validation:**

        >>> sanitizer = dv.TimestampSanitizer(
        ...     data=series,
        ...     nominal_freq='30min',  # Expect 30-minute resolution
        ...     verbose=True
        ... )
        >>> clean_series = sanitizer.get()

        **Selective processing (skip some steps):**

        >>> sanitizer = dv.TimestampSanitizer(
        ...     data=series,
        ...     regularize=False,                    # Keep gaps in data
        ...     output_middle_timestamp=False,       # Keep end-of-period format
        ...     remove_index_nat=True,
        ...     verbose=True
        ... )
        >>> result = sanitizer.get()

        **Error handling for corrupted data:**

        >>> try:
        ...     sanitizer = dv.TimestampSanitizer(
        ...         data=corrupted_data,
        ...         nominal_freq='30min',
        ...         validate_naming=True
        ...     )
        ... except ValueError as e:
        ...     print(f"Timestamp validation failed: {e}")
        ...     # Handle error: fix data or re-run without nominal_freq
        """
        self._validate_input(data)
        self.data = data.copy()
        self.output_middle_timestamp = output_middle_timestamp
        self.validate_naming = validate_naming
        self.convert_to_datetime = convert_to_datetime
        self.remove_index_nat = remove_index_nat
        self.sort_ascending = sort_ascending
        self.remove_duplicates = remove_duplicates
        self.regularize = regularize
        self.nominal_freq = nominal_freq
        self.verbose = verbose

        # Track data modifications for status reporting
        self._original_shape = data.shape
        self._original_rows = len(data)
        self._nat_removed = 0
        self._duplicates_removed = 0
        self._rows_added_by_regularization = 0

        try:
            self.inferred_freq = None if not data.index.freq else data.index.freq
        except AttributeError:
            self.inferred_freq = None

        self._freq_confidence = None  # Confidence in frequency detection (0-1)
        self._freq_detection_method = None  # Which method detected the frequency
        self._freq_percent_matching = None  # % of intervals matching detected frequency
        self._freq_alternatives = []  # Alternative frequencies detected

        self._run()

    def get(self) -> Union[Series, DataFrame]:
        return self.data

    def get_status(self) -> dict:
        """
        Return processing status and summary of changes made.

        Returns a dictionary with information about what was modified during
        timestamp sanitization, useful for understanding data loss or changes.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'original_shape': Original data shape (rows, columns)
            - 'final_shape': Final data shape after processing
            - 'rows_removed': Total rows removed during cleaning
            - 'rows_removed_nat': Rows removed due to NaT values
            - 'rows_removed_duplicates': Duplicate rows removed
            - 'rows_added_by_regularization': Rows added to fill gaps
            - 'net_rows': Final row count minus original row count (can be negative)
            - 'inferred_frequency': Detected time resolution
            - 'frequency_confidence': Confidence score for detected frequency (0.0-1.0)
            - 'frequency_detection_method': Method used to detect frequency
            - 'frequency_percent_matching': % of intervals matching detected frequency
            - 'frequency_alternatives': Alternative frequencies detected by other methods
            - 'timestamp_format': Index name (TIMESTAMP_END, TIMESTAMP_START, or TIMESTAMP_MIDDLE)

        Example
        -------
        >>> sanitizer = dv.TimestampSanitizer(data=df, verbose=False)
        >>> status = sanitizer.get_status()
        >>> print(f"Removed {status['rows_removed']} rows, frequency confidence: {status['frequency_confidence']:.0%}")
        """
        return {
            'original_shape': self._original_shape,
            'final_shape': self.data.shape,
            'rows_removed': self._nat_removed + self._duplicates_removed,
            'rows_removed_nat': self._nat_removed,
            'rows_removed_duplicates': self._duplicates_removed,
            'rows_added_by_regularization': self._rows_added_by_regularization,
            'net_rows': len(self.data) - self._original_rows,
            'inferred_frequency': self.inferred_freq,
            'frequency_confidence': self._freq_confidence,
            'frequency_detection_method': self._freq_detection_method,
            'frequency_percent_matching': self._freq_percent_matching,
            'frequency_alternatives': self._freq_alternatives,
            'timestamp_format': self.data.index.name,
        }

    def _validate_input(self, data: Union[Series, DataFrame]) -> None:
        """Validate input data before processing.

        Raises
        ------
        TypeError
            If data is None or not a Series/DataFrame
        ValueError
            If data is empty or has no valid index
        """
        if data is None:
            raise TypeError("data cannot be None")
        if not isinstance(data, (Series, DataFrame)):
            raise TypeError(f"data must be a Union[Series, DataFrame], got {type(data).__name__}")
        if data.empty:
            raise ValueError("data cannot be empty")
        if data.index is None or len(data.index) == 0:
            raise ValueError("data must have a valid index with at least one element")

    def _run(self):
        if self.verbose:
            rule("Timestamp sanitization", verbose=self.verbose)

        # Validate timestamp name
        if self.validate_naming:
            _ = validate_timestamp_naming(data=self.data, verbose=self.verbose)

        # Convert timestamp to datetime
        if self.convert_to_datetime:
            self.data = convert_timestamp_to_datetime(self.data, verbose=self.verbose)

        # Remove rows that do not have a timestamp
        if self.remove_index_nat:
            self.data, self._nat_removed = remove_rows_nat(df=self.data, verbose=self.verbose)

        # Sort timestamp index ascending
        if self.sort_ascending:
            self.data = sort_timestamp_ascending(self.data, verbose=self.verbose)

        # Remove index duplicates
        if self.remove_duplicates:
            self.data, self._duplicates_removed = remove_index_duplicates(data=self.data, keep='last',
                                                                          verbose=self.verbose)

        # Validate monotonicity (only if sorting was enabled)
        if self.sort_ascending:
            _ = validate_timestamp_monotonic(data=self.data, verbose=self.verbose)

        # Check if data is empty before frequency detection
        if len(self.data) == 0:
            raise ValueError(
                "Data is empty after cleaning (all rows removed by NaT/duplicate removal). "
                "Check your input data: (1) verify timestamps are valid, "
                "(2) try remove_index_nat=False to keep rows with NaT, or "
                "(3) check that no duplicates consumed all rows."
            )

        # Detect time resolution from data
        freq_detector = DetectFrequency(index=self.data.index, verbose=self.verbose)
        detected_freq = freq_detector.get()
        self._freq_confidence = freq_detector.confidence
        self._freq_detection_method = freq_detector.detection_method
        self._freq_percent_matching = freq_detector.percent_matching
        self._freq_alternatives = freq_detector.alternatives

        # If data had pre-existing freq, validate consistency
        if self.inferred_freq and detected_freq != str(self.inferred_freq):
            if self.verbose:
                warn(f"Frequency mismatch ({self.inferred_freq} -> {detected_freq})", verbose=self.verbose)

        # Always use detected frequency as it's more reliable
        self.inferred_freq = detected_freq

        # Compare nominal with detected time resolution
        if self.nominal_freq:
            if self.inferred_freq != self.nominal_freq:
                raise ValueError(
                    f"Timestamp frequency validation failed: detected frequency "
                    f"'{self.inferred_freq}' does not match expected frequency "
                    f"'{self.nominal_freq}'. Either: (1) your data has an incorrect "
                    f"frequency (check for gaps, duplicates, or irregular spacing), "
                    f"or (2) set nominal_freq=None to skip frequency validation."
                )

        # Make timestamp continuous w/o date gaps
        if self.regularize:
            rows_before = len(self.data)
            self.data = continuous_timestamp_freq(data=self.data, freq=self.inferred_freq, verbose=self.verbose)
            self._rows_added_by_regularization = len(self.data) - rows_before

        # Convert timestamp to middle
        if self.output_middle_timestamp:
            self.data = convert_series_timestamp_to_middle(data=self.data, verbose=self.verbose)

        # Check if data is empty after processing
        if len(self.data) == 0:
            raise ValueError(
                "Data is empty after timestamp sanitization. This likely means all rows "
                "were removed (e.g., all timestamps were NaT). Check your input data: "
                "(1) verify timestamps are valid, (2) try remove_index_nat=False to keep "
                "rows with NaT, or (3) check that no other step removed all data."
            )


class DetectFrequency:
    """Detect data time resolution from time series index


    - Example notebook available in:
        notebooks/TimeStamps/Detect_time_resolution.ipynb
    - Unittest:
        test_timestamps.TestTimestamps

    """

    def __init__(self, index: pd.DatetimeIndex, verbose: bool = False):
        self.index = index
        self.verbose = verbose
        # self.freq_expected = freq_expected
        self.num_datarows = self.index.__len__()
        self.freq = None
        self.confidence = None  # Confidence score (0-1)
        self.detection_method = None  # Which method detected the frequency
        self.percent_matching = None  # % of intervals matching detected frequency
        self.alternatives = []  # Alternative frequencies detected by other methods
        self._run()

    def _run(self):
        freq_full, freqinfo_full = timestamp_infer_freq_from_fullset(timestamp_ix=self.index)
        freq_timedelta, freqinfo_timedelta = timestamp_infer_freq_from_timedelta(timestamp_ix=self.index)
        freq_progressive, freqinfo_progressive = timestamp_infer_freq_progressively(timestamp_ix=self.index)

        # Add number to frequency string, needed for Timedelta: e.g. 'min' --> '1min'
        # Check if frequency strings contain a number, b/c Timedelta explicitely needs a number,
        # e.g. '1min' for data in 1-minute time resolution.
        # Note that .infer_freq() that is used in the functions above outputs frequency strings
        # for data with e.g. 1-minute time resolution as `min` (no number), while for higher
        # minute-time resolutions the frequency string contains a number, e.g. '30min'.
        # Same is true for hourly etc... data.
        if freq_full:
            freq_full = f'1{freq_full}' if not any(digit.isdigit() for digit in freq_full) else freq_full
        if freq_timedelta:
            freq_timedelta = f'1{freq_timedelta}' if not any(
                digit.isdigit() for digit in freq_timedelta) else freq_timedelta
        if freq_progressive:
            freq_progressive = f'1{freq_progressive}' if not any(
                digit.isdigit() for digit in freq_progressive) else freq_progressive

        # Harmonize frequency strings
        # This also removes the number in case of e.g. 1-minute time resolution: '1min' --> 'min'.
        # This will yield the frequency string as seen by (the current version of) pandas. The idea
        # is to harmonize between different representations e.g. `T` or `min` for minutes.
        if freq_full:
            freq_full = to_offset(pd.Timedelta(freq_full)).freqstr
        if freq_timedelta:
            freq_timedelta = to_offset(pd.Timedelta(freq_timedelta)).freqstr
        if freq_progressive:
            freq_progressive = to_offset(pd.Timedelta(freq_progressive)).freqstr

        list_of_found_freqs = [freq_full, freq_timedelta, freq_progressive]
        # The all-agree branch must exclude the all-None case: three Nones are
        # "equal", so without this guard undetectable frequencies would silently
        # fall through here instead of reaching the RuntimeError below.
        if list_of_found_freqs[0] is not None and all(i == list_of_found_freqs[0] for i in list_of_found_freqs):
            # if all(f for f in [freq_full, freq_timedelta, freq_progressive]):

            # List of {Set of detected freqs}
            freq_list = list({freq_timedelta, freq_full, freq_progressive})

            if len(freq_list) == 1 and freq_list[0] is not None:
                # Maximum certainty, one single freq found across all checks
                self.freq = freq_list[0]
                self.confidence = 1.0
                self.detection_method = "all_methods_agree"
                # Extract % matching from timedelta if available
                try:
                    conf_str = freqinfo_timedelta.split('%')[0]
                    self.percent_matching = float(conf_str)
                except (ValueError, IndexError):
                    self.percent_matching = 100.0
                if self.verbose:
                    info(f"Detect frequency: {self.freq} (all methods agree)", verbose=self.verbose)

        elif freq_full:
            # High certainty, freq found from full range of dataset
            self.freq = freq_full
            self.confidence = 0.95
            self.detection_method = "full_dataset"
            self.percent_matching = 100.0
            # Track alternatives
            if freq_timedelta:
                self.alternatives.append(freq_timedelta)
            if freq_progressive:
                self.alternatives.append(freq_progressive)
            if self.verbose:
                info(f"Detect frequency: {self.freq} (full data)", verbose=self.verbose)

        elif freq_timedelta:
            # High certainty, freq found from most frequent timestep that
            # occurred at least 90% of the time
            self.freq = freq_timedelta
            self.detection_method = "timedelta"
            # Extract confidence from freqinfo like '75% occurrence'
            try:
                conf_str = freqinfo_timedelta.split('%')[0]
                self.confidence = float(conf_str) / 100.0
                self.percent_matching = float(conf_str)
            except (ValueError, IndexError):
                self.confidence = 0.75
                self.percent_matching = 75.0
            # Track alternatives
            if freq_progressive:
                self.alternatives.append(freq_progressive)
            if self.verbose:
                info(f"Detect frequency: {self.freq} (timedelta, {self.confidence*100:.0f}% match)",
                     verbose=self.verbose)

        elif freq_progressive:
            # Medium certainty, freq found from start and end of dataset
            self.freq = freq_progressive
            self.confidence = 0.70
            self.detection_method = "start_end_chunks"
            self.percent_matching = None  # Not calculated for progressive method
            if self.verbose:
                info(f"Detect frequency: {self.freq} (start/end)", verbose=self.verbose)

        else:
            raise RuntimeError(
                "Could not detect timestamp frequency using any method. This typically "
                "means your timestamps are highly irregular or have too many gaps. "
                "To fix: (1) verify data quality (check for irregular gaps/duplicates), "
                "(2) try regularize=True to fill gaps automatically, or "
                "(3) skip frequency detection with nominal_freq=None."
            )

    def get(self) -> str:
        return self.freq


def format_timestamp_to_fluxnet_format(df: DataFrame, timestamp_col: str) -> Series:
    """
    Convert timestamp column to FLUXNET format (YYYYMMDDhhmm).

    Converts a datetime column in a DataFrame to the standard FLUXNET timestamp
    format as a compressed string without separators. The timestamp must be
    available as a data column (not as the DataFrame index).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the timestamp column.
    timestamp_col : str
        Name of the column containing datetime values to be formatted.

    Returns
    -------
    Series
        Series with timestamps formatted as strings in FLUXNET format (YYYYMMDDhhmm).
        Example: '202307151430' for July 15, 2023 at 14:30.

    Note
    ----
    The timestamp must exist as a data column in the DataFrame, not as the index.
    """
    info(f"Formatting timestamp column {timestamp_col} to %Y%m%d%H%M ...")
    timestamp = df[timestamp_col].dt.strftime('%Y%m%d%H%M')
    return timestamp


def detect_freq_groups(index: DatetimeIndex) -> Series:
    """
    Analyze timestamp for records where the time resolution is absolutely certain

    This function calculates the timedeltas (the time differences) between the current
    timestamp and the timestamp of the record before and after. For data records where
    the two differences are the same (in absolute terms) have an absolutely certain
    timestamp.

    The determined time resolution of each record is described in the newly created column
    'FREQ_AUTO_SEC' in terms of seconds. The column is added to *df* and can be used to
    access the different time resolution groups separately during later processing.

    Example:

                        TIMESTAMP_CURRENT   TIMESTAMP_PREV      TIMESTAMP_NEXT           DELTA_PREV  DELTA_NEXT  DELTA_DIFF
    TIMESTAMP_CURRENT
    2020-10-01 00:20:00 2020-10-01 00:20:00 2020-10-01 00:10:00 2020-10-01 00:30:00      -600.0       600.0         0.0
    2020-10-01 00:30:00 2020-10-01 00:30:00 2020-10-01 00:20:00 2020-10-01 00:40:00      -600.0       600.0         0.0
    2020-10-01 00:40:00 2020-10-01 00:40:00 2020-10-01 00:30:00 2020-10-01 00:50:00      -600.0       600.0         0.0
    ...                                 ...                 ...                 ...         ...         ...         ...
    2021-09-30 23:57:00 2021-09-30 23:57:00 2021-09-30 23:56:00 2021-09-30 23:58:00       -60.0        60.0         0.0
    2021-09-30 23:58:00 2021-09-30 23:58:00 2021-09-30 23:57:00 2021-09-30 23:59:00       -60.0        60.0         0.0
    2021-09-30 23:59:00 2021-09-30 23:59:00 2021-09-30 23:58:00 2021-10-01 00:00:00       -60.0        60.0         0.0

    The example shows a dataset that starts with 10MIN time resolution and ends with
    1MIN time resolution. For each record, the previous and next timestamp are detected
    and the delta between these two and the current timestamp are calculated. The sum
    of DELTA_PREV and DELTA_NEXT will yield DELTA_DIFF = 0 if the time differences
    were the same. DELTA_DIFF = 0 therefore describes locations where the time resolution
    is certain.

    For 10MIN time records, the 'FREQ_AUTO_SEC' will be set to '600', for 1MIN records
    to '60'. The first and last records of each frequency group are added during processing,
    e.g., the timestamp '2020-10-01 00:10:00' is the first timestamp for the '600' group
    although it is not part of the main index (main index starts one record later with
    '2020-10-01 00:20:00' because '2020-10-01 00:10:00' does not have a value for
    TIMESTAMP_PREV).

    Note:
        Sometimes there are transition periods between one time resolution and another.
        For example, when the time resolution changes from 10MIN to 1MIN, there might be
        several records in between that have neither 10MIN nor 1MIN, but e.g. 7S or 29MIN etc.
        For these transitional records, the time resolution is not clear and therefore they
        are discarded. This typically affects only a handful of records during the transition
        period(s).

    Args:
        index: Time series dataframe

    Returns:
        df: Time series dataframe with the new column 'FREQ_AUTO_SEC' added

    :: Added in v0.43.0
    """

    groups_ser = pd.Series(index=index, data=np.nan, name='FREQ_AUTO_SEC')
    # index['FREQ_AUTO_SEC'] = np.nan

    # Analyse data for different time resolutions
    timedeltas_df = pd.DataFrame()
    timedeltas_df['TIMESTAMP_CURRENT'] = index

    # Add previous and next timestamps
    timedeltas_df['TIMESTAMP_PREV'] = timedeltas_df['TIMESTAMP_CURRENT'].shift(1)
    timedeltas_df['TIMESTAMP_NEXT'] = timedeltas_df['TIMESTAMP_CURRENT'].shift(-1)

    # DELTA is the difference between the current and the previous/next timestamp,
    # expressed as total seconds
    timedeltas_df['DELTA_PREV'] = timedeltas_df['TIMESTAMP_PREV'].sub(timedeltas_df['TIMESTAMP_CURRENT'])
    timedeltas_df['DELTA_NEXT'] = timedeltas_df['TIMESTAMP_NEXT'].sub(timedeltas_df['TIMESTAMP_CURRENT'])
    timedeltas_df['DELTA_PREV'] = timedeltas_df['DELTA_PREV'].dt.total_seconds()
    timedeltas_df['DELTA_NEXT'] = timedeltas_df['DELTA_NEXT'].dt.total_seconds()

    # The sum of DELTA_PREV and DELTA_NEXT can identify data records where
    # the time resolution is unambiguous.
    # For example: DELTA_PREV = -60, DELTA_NEXT = +60, DELTA_DIFF = 0
    #   In this case the time differences of the current timestamp to
    #   the previous and next timestamps are the same (in absolute terms)
    #   and therefore yields the sum zero.
    timedeltas_df['DELTA_DIFF'] = timedeltas_df['DELTA_PREV'] + timedeltas_df['DELTA_NEXT']
    ix = timedeltas_df['DELTA_DIFF'] == 0
    timedelta_unambiguous_df = timedeltas_df.loc[ix].copy()
    timedelta_unambiguous_df = timedelta_unambiguous_df.set_index(timedelta_unambiguous_df['TIMESTAMP_CURRENT'])

    # Count occurrences of respective DELTA
    delta_counts_df = timedelta_unambiguous_df['DELTA_NEXT'].groupby(
        timedelta_unambiguous_df['DELTA_NEXT']).count().sort_values(ascending=False)
    delta_counts_df = pd.DataFrame(delta_counts_df)
    delta_counts_df = delta_counts_df.rename(columns={"DELTA_NEXT": "COUNTS"})

    # Calculate how much time is covered by each DELTA
    delta_counts_df['DELTA_NEXT'] = delta_counts_df.index
    delta_counts_df['DELTA_TOTAL_TIME'] = delta_counts_df['DELTA_NEXT'].multiply(delta_counts_df['COUNTS'])
    delta_counts_df['TOTAL_TIME'] = delta_counts_df['DELTA_TOTAL_TIME'].sum()
    delta_counts_df['%_DELTA_TOTAL_TIME'] = delta_counts_df['DELTA_TOTAL_TIME'] / delta_counts_df['TOTAL_TIME']
    delta_counts_df['%_DELTA_TOTAL_TIME'] = delta_counts_df['%_DELTA_TOTAL_TIME'] * 100

    # List of found time resolutions (unambiguous)
    deltas = delta_counts_df['DELTA_NEXT'].to_list()

    # Detect first and last date for each delta
    # First and last dates need to be included by using:
    #   - 'TIMESTAMP_PREV' for first date
    #   - 'TIMESTAMP_NEXT' for last date
    for d in deltas:
        this_delta = timedelta_unambiguous_df.loc[timedelta_unambiguous_df['DELTA_NEXT'] == d].copy()
        this_delta = this_delta.set_index(this_delta['TIMESTAMP_CURRENT'])
        first_date = this_delta['TIMESTAMP_PREV'].min()
        last_date = this_delta['TIMESTAMP_NEXT'].max()

        # Add first and last date to df
        new_index = this_delta.index.union([first_date, last_date])
        this_delta = this_delta.reindex(new_index)

        groups_ser.loc[this_delta.index] = d

        # freq = f"{int(d)}S"
        # _index = pd.date_range(start=first_date, end=last_date, freq=freq)
        # this_delta.reindex(_index)
        delta_counts_df.loc[d, 'FIRST_DATE'] = first_date
        delta_counts_df.loc[d, 'LAST_DATE'] = last_date

    return groups_ser


def sort_timestamp_ascending(data: Union[Series, DataFrame], verbose: bool = False) -> Union[Series, DataFrame]:
    """
    Sort timestamp index in ascending order.

    Reorders data rows so that timestamps increase from first to last row.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index to be sorted.
    verbose : bool, optional
        Print status message. Default is False.

    Returns
    -------
    Union[Series, DataFrame]
        Data with sorted timestamp index.

    Examples
    --------
    >>> df_sorted = sort_timestamp_ascending(df, verbose=True)
    """
    if verbose:
        info("Sort ascending: OK", verbose=verbose)
    data = data.sort_index()
    return data


def remove_rows_nat(df: Union[Series, DataFrame], verbose: bool = False) -> tuple[Union[Series, DataFrame], int]:
    """
    Remove rows that do not have a timestamp (NaT).

    Identifies and removes any rows with missing timestamp values (NaT) in the index.
    Issues a warning if more than 10% of rows are removed.

    Parameters
    ----------
    df : Union[Series, DataFrame]
        Data with timestamp index.
    verbose : bool, optional
        Print status messages. Default is False.

    Returns
    -------
    tuple[Union[Series, DataFrame], int]
        - Union[Series, DataFrame]: Data with NaT rows removed
        - int: Number of rows removed

    Raises
    ------
    ValueError
        If all rows are removed (all timestamps are NaT).

    Examples
    --------
    >>> df_clean, n_removed = remove_rows_nat(df, verbose=True)
    >>> print(f"Removed {n_removed} NaT rows")
    """
    no_date = df.index.isnull()
    n_rows = no_date.sum()
    original_length = len(df)
    if n_rows > 0:
        df = df.loc[df.index[~no_date]].copy()
        pct_removed = 100 * n_rows / original_length
        if verbose:
            if pct_removed > 10:
                warn(f"Remove NaT values: {n_rows} removed [{pct_removed:.1f}% of data]", verbose=verbose)
            else:
                info(f"Remove NaT values: {n_rows} removed", verbose=verbose)
    else:
        if verbose:
            info("Remove NaT values: none", verbose=verbose)
    return df, n_rows


def convert_timestamp_to_datetime(data: Union[Series, DataFrame], verbose: bool = False) -> Union[Series, DataFrame]:
    """
    Convert timestamp index to datetime format.

    Ensures the timestamp index is in pandas DatetimeIndex format. This is an
    important validation step since data may have timestamps as strings or other
    types. Uses pandas.to_datetime with coercion to handle various timestamp formats.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index to be converted.
    verbose : bool, optional
        Print status message. Default is False.

    Returns
    -------
    Union[Series, DataFrame]
        Data with DatetimeIndex timestamp index.

    Raises
    ------
    ValueError
        If timestamp index cannot be converted to datetime format.

    Examples
    --------
    >>> df_dt = convert_timestamp_to_datetime(df, verbose=True)
    """
    try:
        data.index = pd.to_datetime(data.index, errors='coerce')
        if verbose:
            info("Convert to datetime: OK", verbose=verbose)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to convert timestamp to datetime format. "
            f"Original error: {e}"
        ) from e
    return data


def validate_timestamp_naming(data: Union[Series, DataFrame], verbose: bool = False) -> str:
    """
    Check if timestamp is correctly named.

    Validates that the timestamp index has a required name indicating whether it
    refers to the START, MIDDLE, or END of the averaging period. This is important
    for correct data aggregation and interpretation.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index to be validated.
    verbose : bool, optional
        Print status message. Default is False.

    Returns
    -------
    str
        The validated timestamp index name ('TIMESTAMP_END', 'TIMESTAMP_START',
        or 'TIMESTAMP_MIDDLE').

    Raises
    ------
    ValueError
        If timestamp index has no name or name is not one of the allowed values.

    Notes
    -----
    Allowed names are:
    - 'TIMESTAMP_END': Timestamp marks the END of the averaging period
    - 'TIMESTAMP_START': Timestamp marks the START of the averaging period
    - 'TIMESTAMP_MIDDLE': Timestamp marks the MIDDLE of the averaging period

    Examples
    --------
    >>> df.index.name = 'TIMESTAMP_END'
    >>> name = validate_timestamp_naming(df, verbose=True)
    >>> print(f"Valid timestamp format: {name}")
    """
    timestamp_name = data.index.name
    allowed_timestamp_names = ['TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE']
    # Check if timestamp name is None
    if timestamp_name is None:
        raise ValueError(
            "Timestamp index has no name. Expected one of "
            f"{allowed_timestamp_names}. Set the index name to indicate "
            "whether the timestamp represents END, START, or MIDDLE of the "
            "averaging period. Example: data.index.name = 'TIMESTAMP_END'"
        )

    # First check if timestamp already has one of the required names
    if any(fnmatch.fnmatch(timestamp_name, allowed_name) for allowed_name in allowed_timestamp_names):
        if verbose:
            info(f"Validate naming ({timestamp_name}): OK", verbose=verbose)
        return timestamp_name

    else:
        raise ValueError(
            f"Timestamp index naming validation failed: index name is '{timestamp_name}' "
            f"but must be one of {allowed_timestamp_names}. This indicates whether the "
            f"timestamp refers to the END, START, or MIDDLE of the averaging period. "
            f"Either: (1) rename your index before passing to TimestampSanitizer "
            f"(e.g., data.index.name = 'TIMESTAMP_END'), or (2) set validate_naming=False "
            f"if you've already validated the naming yourself."
        )


def validate_timestamp_monotonic(data: Union[Series, DataFrame], verbose: bool = False) -> None:
    """
    Validate that timestamp index is strictly monotonic increasing.

    Ensures timestamps are in strict ascending order with no duplicates or
    backward time jumps. This check should occur after sorting and duplicate
    removal to catch any remaining issues before frequency detection.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index to be validated.
    verbose : bool, optional
        Print status message. Default is False.

    Raises
    ------
    ValueError
        If timestamp index is not strictly monotonic increasing.

    Notes
    -----
    "Strictly monotonic" means no two timestamps are equal and all are in
    ascending order. After sorting and duplicate removal, data should always
    pass this check. If it fails, it indicates a data or processing error.

    Examples
    --------
    >>> validate_timestamp_monotonic(df, verbose=True)
    """
    if not data.index.is_monotonic_increasing:
        # Find where monotonicity breaks to help debugging
        diffs = data.index.to_series().diff()
        backward_jumps = diffs[diffs <= pd.Timedelta(0)]
        n_issues = len(backward_jumps)
        first_issue_idx = backward_jumps.index[0] if len(backward_jumps) > 0 else None

        raise ValueError(
            f"Timestamp index is not strictly monotonic increasing. Found {n_issues} "
            f"non-increasing intervals. First issue at index position "
            f"{data.index.get_loc(first_issue_idx) if first_issue_idx else '?'}: "
            f"{first_issue_idx}. This should not occur after sorting and duplicate "
            f"removal. Check for: (1) remaining duplicates with identical timestamps, "
            f"(2) data from multiple sources with conflicting times, or "
            f"(3) sorting/duplicate removal errors."
        )

    if verbose:
        info("Validate monotonicity: OK", verbose=verbose)


def current_unixtime() -> int:
    """
    Current time as integer number of nanoseconds since the epoch

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    current_time_unix = time.time_ns()
    return current_time_unix


def current_datetime(str_format: str = '%Y-%m-%d %H:%M:%S') -> tuple[dt.datetime, str]:
    """
    Current datetime as datetime and string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime(str_format)
    return now_time_dt, now_time_str


def current_date_str_condensed() -> str:
    """
    Current date as string

    - Example notebook available in:
        -
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_datetime_str_condensed() -> str:
    """
    Current datetime as string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d%H%M%S")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_time_microseconds_str() -> str:
    """
    Current time including microseconds as string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def make_run_id(prefix: str = False) -> str:
    """
    Create string identifier that includes current datetime

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt, _ = current_datetime()
    now_time_str = now_time_dt.strftime("%Y%m%d-%H%M%S")
    prefix = prefix if prefix else "RUN"
    run_id = f"{prefix}-{now_time_str}"
    return run_id


# def timedelta_to_string(timedelta):
#     """
#     Converts a pandas.Timedelta to a frequency string representation
#     compatible with pandas.Timedelta constructor format
#     https://stackoverflow.com/questions/46429736/pandas-resampling-how-to-generate-offset-rules-string-from-timedelta
#     https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
#
#     This function is part of diive v0.67.0.
#
#     """
#     c = timedelta.components
#     format = ''
#     if c.days != 0:
#         format += '%dD' % c.days
#     if c.hours > 0:
#         format += '%dh' % c.hours
#     if c.minutes > 0:
#         format += '%dmin' % c.minutes
#     if c.seconds > 0:
#         format += '%ds' % c.seconds
#     if c.milliseconds > 0:
#         format += '%dms' % c.milliseconds
#         # old format: format += '%dL' % c.milliseconds
#     if c.microseconds > 0:
#         format += '%dus' % c.microseconds
#         # old format: format += '%dU' % c.microseconds
#     if c.nanoseconds > 0:
#         format += '%dns' % c.nanoseconds
#         # old format: format += '%dN' % c.nanoseconds
#
#     # Remove leading `1` to represent e.g. daily resolution
#     # This is in line with how pandas handles frequency strings,
#     # e.g., 1-minute time resolution is represented by `min` and
#     # not by `1min`.
#     if format == '1D':
#         format = 'D'
#     elif format == '1h':
#         format = 'h'
#     elif format == '1min':
#         format = 'min'
#     elif format == '1s':
#         format = 's'
#     elif format == '1ms':
#         format = 'ms'
#     elif format == '1us':
#         format = 'us'
#     elif format == '1ns':
#         format = 'ns'
#
#     return format


def generate_freq_timedelta_from_freq(to_duration, to_freq):
    """
    Generate timedelta with given duration and frequency

    Does not really work with M or Y frequency b/c of their different number of days,
    e.g. August 31 days but September has 30 days.

    The Timedelta can be directly used in operations, e.g. when one single timestamp
    entry is available and it is needed to calculate the previous timestamp. With the
    Timedelta, the previous timestamp can be calculated by simply subtracting the
    Timedelta from the available timestamp.

    Example:
        >> to_duration = 1
        >> to_freq = 'D'
        >> pd.to_timedelta(to_duration, unit=to_freq)
        Timedelta('1 days 00:00:00')

    :param to_duration: int
    :param to_freq: pandas frequency string
                    see here for options:
                    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :return: Timedelta
    """
    timedelta = pd.to_timedelta(to_duration, unit=to_freq)
    return timedelta


def build_timestamp_range(start_dt, df_len, freq):
    """ Builds timestamp column starting with start date and
        the given frequency.

    :param df_len: int (number of rows)
    :param freq: pandas freq string (e.g. '1s' for 1 second steps)
    :return:
    """

    add_timedelta = (df_len - 1) * pd.to_timedelta(freq)
    end_dt = start_dt + pd.Timedelta(add_timedelta)
    date_rng = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    return date_rng


def vectorize_timestamps(df,
                         year: bool = True,
                         season: bool = True,
                         month: bool = True,
                         week: bool = True,
                         doy: bool = True,
                         hour: bool = True,
                         txt: str = "",
                         verbose: int = 1) -> DataFrame:
    """
    Vectorizes a DatetimeIndex into linear and cyclical (sin/cos) numerical features.

    This function "vectorizes" time by mapping periodic date components (like month or
    hour) into 2D space using sine and cosine transformations. This ensures that
    cyclical proximity (e.g., December and January, or 23:00 and 00:00) is preserved
    geometrically for machine learning models.

    Kudos:
    - https://datascience.stackexchange.com/questions/60951/is-it-necessary-to-convert-labels-in-string-to-integer-for-scikit-learn-and-xgbo
    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    Args:
        df (DataFrame): Input DataFrame with a DatetimeIndex.
        year (bool): Add linear year column.
        season (bool): Add season (1-4) and sin/cos columns.
        month (bool): Add month (1-12) and sin/cos columns.
        week (bool): Add week (1-53) and sin/cos columns.
        doy (bool): Add day of year (1-366) and sin/cos columns.
        hour (bool): Add hour (0-23) and sin/cos columns.
        txt (str): Optional suffix text for the verbose print statement.
        verbose (int): If > 0, prints the list of added columns.

    Returns:
        DataFrame: A copy of the dataframe with the new columns appended.
    """
    df = df.copy()
    newcols = []

    year_col = '.YEAR'
    month_col = '.MONTH'
    week_col = '.WEEK'
    doy_col = '.DOY'
    hour_col = '.HOUR'
    season_col = '.SEASON'

    if year:
        # Not cyclical
        newcols.append(year_col)
        df[year_col] = df.index.year.astype(int)

    if season:
        # Cyclical
        season_sin_col = f'{season_col}_SIN'
        season_cos_col = f'{season_col}_COS'
        season_period = 4
        df[season_col] = insert_season(timestamp=df.index)
        # Cast to float to ensure the division works smoothly even with Int64 types
        df[season_sin_col] = np.sin(2 * np.pi * df[season_col].astype(float) / season_period)
        df[season_cos_col] = np.cos(2 * np.pi * df[season_col].astype(float) / season_period)
        newcols += [season_col, season_sin_col, season_cos_col]

    if month:
        # Cyclical
        month_sin_col = f'{month_col}_SIN'
        month_cos_col = f'{month_col}_COS'
        month_period = 12
        df[month_col] = df.index.month.astype(int)
        df[month_sin_col] = np.sin(2 * np.pi * df[month_col] / month_period)
        df[month_cos_col] = np.cos(2 * np.pi * df[month_col] / month_period)
        newcols += [month_col, month_sin_col, month_cos_col]

    if week:
        # Cyclical
        # Use 53 weeks for week_period because ISO years can have up to 53 weeks.
        # This ensures week 53 maps smoothly back toward week 1
        week_sin_col = f'{week_col}_SIN'
        week_cos_col = f'{week_col}_COS'
        week_period = 53
        df[week_col] = df.index.isocalendar().week.astype(int)
        df[week_sin_col] = np.sin(2 * np.pi * df[week_col] / week_period)
        df[week_cos_col] = np.cos(2 * np.pi * df[week_col] / week_period)
        newcols += [week_col, week_sin_col, week_cos_col]

    if doy:
        # Cyclical
        doy_sin_col = f'{doy_col}_SIN'
        doy_cos_col = f'{doy_col}_COS'
        year_period = 365.25  # Approx. days in a year
        df[doy_col] = df.index.dayofyear.astype(int)
        df[doy_sin_col] = np.sin(2 * np.pi * df[doy_col] / year_period)
        df[doy_cos_col] = np.cos(2 * np.pi * df[doy_col] / year_period)
        newcols += [doy_col, doy_sin_col, doy_cos_col]

    if hour:
        # Cyclical
        hour_sin_col = f'{hour_col}_SIN'
        hour_cos_col = f'{hour_col}_COS'
        hour_period = 24
        df[hour_col] = df.index.hour.astype(int)
        df[hour_sin_col] = np.sin(2 * np.pi * df[hour_col] / hour_period)
        df[hour_cos_col] = np.cos(2 * np.pi * df[hour_col] / hour_period)
        newcols += [hour_col, hour_sin_col, hour_cos_col]

    # Year + Month (Format: YYYYMM, e.g., 202308)
    if year and month:
        yearmonth_col = '.YEARMONTH'
        newcols.append(yearmonth_col)
        # Math is faster and cleaner than string conversion
        df[yearmonth_col] = df[year_col] * 100 + df[month_col]

    # Year + DOY (Format: YYYYDDD, e.g., 2023194)
    if year and doy:
        yeardoy_col = '.YEARDOY'
        newcols.append(yeardoy_col)
        # Multiply by 1000 to accommodate up to 366 days
        df[yeardoy_col] = df[year_col] * 1000 + df[doy_col]

    # Year + Week (Format: YYYYWW, e.g., 202315)
    if year and week:
        yearweek_col = '.YEARWEEK'
        newcols.append(yearweek_col)
        # Multiply by 100 to accommodate up to 53 weeks
        df[yearweek_col] = df[year_col] * 100 + df[week_col]

    if verbose > 0:
        info(f"++ Added new columns with timestamp info: {newcols} {txt}")

    return df


def insert_season(
        timestamp: DatetimeIndex,
        spring: Optional[List[int]] = None,
        summer: Optional[List[int]] = None,
        autumn: Optional[List[int]] = None,
        winter: Optional[List[int]] = None,
) -> Index:
    """
    Inserts seasonal labels into a pandas Timestamp index based on a customizable mapping
    of months to seasons. The function allows for flexible assignment of months to seasons
    and verifies robustness against overlapping or incomplete season definitions.

    Args:
        timestamp (DatetimeIndex): The pandas DatetimeIndex for which seasonal labels
            are assigned based on the provided mapping.
        spring (Optional[List[int]]): A list of month numbers (1 to 12) defining spring.
            Defaults to a predefined season map if not provided.
        summer (Optional[List[int]]): A list of month numbers (1 to 12) defining summer.
            Defaults to a predefined season map if not provided.
        autumn (Optional[List[int]]): A list of month numbers (1 to 12) defining autumn.
            Defaults to a predefined season map if not provided.
        winter (Optional[List[int]]): A list of month numbers (1 to 12) defining winter.
            Defaults to a predefined season map if not provided.

    Returns:
        Index: A pandas Index object with seasonal labels (as nullable integer dtype)
            corresponding to the input DatetimeIndex. NaN values are returned for months
            not mapped to any season.

    Raises:
        ValueError: If a month is assigned to multiple seasons, resulting in overlapping
            definitions.
    """
    # Handle Mutable Defaults
    season_months = {
        1: spring if spring is not None else DEFAULT_SEASON_MAP[1],
        2: summer if summer is not None else DEFAULT_SEASON_MAP[2],
        3: autumn if autumn is not None else DEFAULT_SEASON_MAP[3],
        4: winter if winter is not None else DEFAULT_SEASON_MAP[4],
    }

    # Robustness Check: ensure no month is assigned to more than one season
    all_months = [m for sublist in season_months.values() for m in sublist]
    if len(all_months) != len(set(all_months)):
        raise ValueError("Season definitions overlap: A month is assigned to multiple seasons.")
    if len(all_months) != 12:
        warn("Not all 12 months are defined in the seasons. Unassigned months will be set to NaN.")

    # Create a single Month-to-Season mapping dictionary
    # Example: {3: 1, 4: 1, 5: 1, 6: 2, ...}
    month_to_season_map = {}
    for season_id, month_list in season_months.items():
        for month in month_list:
            month_to_season_map[month] = season_id

    # Extract the month numbers from the DatetimeIndex
    month_series = timestamp.month

    # Use .map() for assignment operation
    # The default return value for unmapped months is NaN
    season_series = month_series.map(month_to_season_map)

    # Convert the resulting Series back to Int64Dtype (pandas' nullable integer) to safely handle potential NaNs
    return season_series.astype('Int64')


def timestamp_infer_freq_progressively(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """
    Infer frequency by comparing first and last intervals of timestamp index.

    Checks if the first N rows and last N rows have the same frequency, starting
    with N=1000 and progressively reducing to N=3. If both match, the frequency is
    consistent throughout the dataset. This method is fast and robust for detecting
    regular sampling patterns even in large datasets.

    Parameters
    ----------
    timestamp_ix : pd.DatetimeIndex
        Timestamp index to analyze.

    Returns
    -------
    tuple
        (inferred_freq, freqinfo)
        - inferred_freq : str or None
            Detected frequency string (e.g., '30min', '1h'), or None if detection failed.
        - freqinfo : str or None
            Detection method description ('data N+N' if successful, None otherwise).

    Notes
    -----
    - Requires at least 6 rows (3 from start, 3 from end) for detection
    - Returns frequency only if start and end intervals match
    - Useful for quick frequency validation before full dataset analysis
    """
    MAX_CHECK_RANGE = 1000  # Start with up to 1000 rows from each end
    MIN_CHECK_RANGE = 3  # Minimum rows needed for frequency detection

    n_datarows = timestamp_ix.__len__()
    inferred_freq = None
    freqinfo = None

    if n_datarows > 0:
        for ndr in range(MAX_CHECK_RANGE, MIN_CHECK_RANGE, -1):
            if n_datarows >= ndr * 2:  # Same amount of ndr needed for start and end of file
                _inferred_freq_start = pd.infer_freq(timestamp_ix[0:ndr])
                _inferred_freq_end = pd.infer_freq(timestamp_ix[-ndr:])
                inferred_freq = _inferred_freq_start if _inferred_freq_start == _inferred_freq_end else None
                if inferred_freq:
                    freqinfo = f'data {ndr}+{ndr}' if inferred_freq else '-'
                    return inferred_freq, freqinfo
            else:
                continue
    return inferred_freq, freqinfo


def timestamp_infer_freq_from_fullset(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """
    Infer frequency from the complete timestamp index using pandas inference.

    Analyzes all timestamps to detect regular sampling patterns. This method uses
    pandas' built-in frequency inference and is most reliable for strictly regular
    data without gaps or irregular intervals.

    Parameters
    ----------
    timestamp_ix : pd.DatetimeIndex
        Timestamp index to analyze.

    Returns
    -------
    tuple
        (inferred_freq, freqinfo)
        - inferred_freq : str or None
            Detected frequency string (e.g., '30min', '1h'), or None if detection failed.
        - freqinfo : str
            Detection result ('full data' if successful, '-not-enough-datarows-' if <10 rows,
            '-failed-' if inference failed).

    Notes
    -----
    - Requires at least 10 timestamps for analysis
    - Most reliable for perfectly regular, gap-free data
    - Returns None if data has irregular intervals or gaps
    - Use in combination with other methods for robust detection
    """
    inferred_freq = None
    freqinfo = None
    n_datarows = timestamp_ix.__len__()
    if n_datarows < 10:
        freqinfo = '-not-enough-datarows-'
        return inferred_freq, freqinfo
    inferred_freq = pd.infer_freq(timestamp_ix)
    if inferred_freq:
        freqinfo = 'full data'
        return inferred_freq, freqinfo
    else:
        freqinfo = '-failed-'
        return inferred_freq, freqinfo


def timestamp_infer_freq_from_timedelta(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """
    Infer frequency from the most common interval between successive timestamps.

    Calculates differences between consecutive timestamps and identifies the most
    frequent interval. This method is robust to minor irregularities and works even
    when data has small gaps or occasional interval variations, as long as one
    interval dominates (>50% of observations).

    Parameters
    ----------
    timestamp_ix : pd.DatetimeIndex
        Timestamp index to analyze.

    Returns
    -------
    tuple
        (inferred_freq, freqinfo)
        - inferred_freq : str or None
            Detected frequency string (e.g., '30min', '1h'), or None if no interval
            appears in >50% of data.
        - freqinfo : str
            Detection result with statistics (e.g., 'timedelta 99.5%'),
            or '-failed-' if detection failed.

    Notes
    -----
    - Requires at least 2 timestamps (to calculate one interval)
    - Robust to occasional irregular intervals
    - Returns frequency only if most common interval covers >50% of all intervals
    - Useful for data with small gaps or timing variations

    References
    ----------
    - https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
    - https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
    """
    inferred_freq = None
    freqinfo = None
    df = pd.DataFrame(columns=['tvalue'])
    df['tvalue'] = timestamp_ix
    df['tvalue_shifted'] = df['tvalue'].shift()
    df['delta'] = (df['tvalue'] - df['tvalue_shifted'])
    n_rows = df['delta'].size  # Total length of data
    detected_deltas = df['delta'].value_counts()  # Found unique deltas
    most_frequent_delta = df['delta'].mode()[0]  # Delta with most occurrences
    most_frequent_delta_counts = detected_deltas[
        most_frequent_delta]  # Number of occurrences for most frequent delta
    most_frequent_delta_perc = most_frequent_delta_counts / n_rows  # Fraction
    # Check whether the most frequent delta appears in >50% of all data rows
    if most_frequent_delta_perc > 0.50:
        inferred_freq = to_offset(most_frequent_delta)
        inferred_freq = inferred_freq.freqstr
        # inferred_freq = timedelta_to_string(most_frequent_delta)
        freqinfo = f'{most_frequent_delta_perc * 100:.0f}% occurrence'
        # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
        return inferred_freq, freqinfo
    # if most_frequent_delta_perc > 0.90:
    #     inferred_freq = to_offset(most_frequent_delta)
    #     inferred_freq = inferred_freq.freqstr
    #     # inferred_freq = timedelta_to_string(most_frequent_delta)
    #     freqinfo = '>90% occurrence'
    #     # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
    #     return inferred_freq, freqinfo
    else:
        freqinfo = '-failed-'
        return inferred_freq, freqinfo


def remove_index_duplicates(data: Union[Series, DataFrame],
                            keep: Literal["first", "last", False] = "last",
                            verbose: bool = False) -> tuple[Union[Series, DataFrame], int]:
    """
    Remove index duplicates.

    Identifies and removes duplicate timestamp entries, keeping the first or last
    occurrence as specified. Issues a warning if more than 10% of rows are duplicates.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index.
    keep : {'first', 'last'}, optional
        Which duplicate to keep. 'first' keeps the first occurrence, 'last' keeps the
        last (default).
    verbose : bool, optional
        Print status messages. Default is False.

    Returns
    -------
    tuple[Union[Series, DataFrame], int]
        - Union[Series, DataFrame]: Data with duplicates removed
        - int: Number of duplicate rows removed

    Examples
    --------
    >>> df_clean, n_removed = remove_index_duplicates(df, keep='last', verbose=True)
    >>> print(f"Removed {n_removed} duplicate timestamps")

    Notes
    -----
    Duplicate detection is based on the index only, not data values.
    """
    n_duplicates = data.index.duplicated().sum()
    if verbose:
        if n_duplicates > 0:
            pct_removed = 100 * n_duplicates / len(data)
            if pct_removed > 10:
                warn(f"Remove duplicates: {n_duplicates} removed [{pct_removed:.1f}% of data]", verbose=verbose)
            else:
                info(f"Remove duplicates: {n_duplicates} removed", verbose=verbose)
        else:
            info("Remove duplicates: none", verbose=verbose)

    if n_duplicates > 0:
        data = data[~data.index.duplicated(keep=keep)]

    return data, n_duplicates


def continuous_timestamp_freq(data: Union[Series, DataFrame], freq: str, verbose: bool = False) -> Union[
    Series, DataFrame]:
    """
    Generate continuous timestamp index without gaps.

    Creates a regular timestamp grid from first to last date at the specified frequency,
    then reindexes data to this grid. Gaps in the original data are filled with NaN
    rows. Useful for time series analysis that requires uniform timesteps.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index.
    freq : str
        Target frequency string (e.g., '30min', '1h', '1D'). Must be a valid pandas
        frequency string.
    verbose : bool, optional
        Print status message. Default is False.

    Returns
    -------
    Union[Series, DataFrame]
        Data with continuous timestamp index at specified frequency. Rows added to
        fill gaps will have NaN values.

    Raises
    ------
    ValueError
        If freq is not a valid pandas frequency string.

    Notes
    -----
    - Original timestamp index name is preserved
    - Data values are NaN for any timestamps not in original data
    - The number of rows will increase if there are gaps in the original data

    Examples
    --------
    >>> df_continuous = continuous_timestamp_freq(df, freq='30min', verbose=True)
    """
    first_date = data.index[0]
    last_date = data.index[-1]

    # Original timestamp name
    idx_name = data.index.name

    # Generate timestamp index b/w first and last date
    _index = pd.date_range(start=first_date, end=last_date, freq=freq)

    data = data.reindex(_index)
    data.index.name = idx_name

    # Set freq
    data.index = pd.to_datetime(data.index)
    data = data.asfreq(freq=freq)
    if verbose:
        info("Regularize gaps: OK", verbose=verbose)
    return data


def insert_timestamp(
        data: DataFrame or Series,
        convention: Literal['start', 'middle', 'end'],
        insert_as_first_col: bool = True,
        verbose: bool = False,
        set_as_index: bool = False) -> DataFrame:
    """
    Insert timestamp column that shows the START, END or MIDDLE time of the averaging interval

    The new timestamp column is added as data column, the current
    *data* index remains unchanged.

    The current *data* index must be a properly named timestamp index.
    Allowed names are: 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE', 'TIMESTAMP_END'.

    Args:
        data: Dataset to which the new timestamp is added as new column
        convention: Timestamp convention of the new timestamp column
            - 'start': Timestamp denoting start of averaging interval
            - 'middle': Timestamp denoting middle of averaging interval
            - 'end': Timestamp denoting end of averaging interval
        insert_as_first_col: If *True*, the new timestamp column is
            added as the first column to *data*. If *False*, the new
            timestamp column is added as the last column to *data*.
        set_as_index: Sets the new timestamp column as dataframe index.
        verbose: If *True*, gives additional text output

    Returns:
        *data* with newly added timestamp column

    Added in: v0.52.0
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    # Current index timestamp name
    timestamp_index_name = data.index.name

    # Check if current index timestamp properly named
    allowed_timestamp_names = ['TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE']
    if timestamp_index_name not in allowed_timestamp_names:
        raise ValueError(
            f"Invalid timestamp index name: '{timestamp_index_name}'. "
            f"Expected one of {allowed_timestamp_names}. The index name indicates "
            f"whether the timestamp represents the END, START, or MIDDLE of the "
            f"averaging period. Rename your index before calling this function."
        )

    # Name of new timestamp series
    new_timestamp_col = None
    if convention == 'start':
        new_timestamp_col = 'TIMESTAMP_START'
    elif convention == 'middle':
        new_timestamp_col = 'TIMESTAMP_MIDDLE'
    elif convention == 'end':
        new_timestamp_col = 'TIMESTAMP_END'

    # Get time resolution of data
    timestamp_freq = data.index.freq
    timestamp_freqstr = data.index.freqstr

    if verbose:
        info(f"Adding new timestamp column {new_timestamp_col} to show {convention} of averaging period ...",
             verbose=verbose)

    # Interval of data records
    timedelta = pd.to_timedelta(timestamp_freq)
    timedelta_half = timedelta / 2

    # Data has MIDDLE timestamp
    if timestamp_index_name == 'TIMESTAMP_MIDDLE':
        if new_timestamp_col == 'TIMESTAMP_MIDDLE':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_END':
            # '2023-03-05 18:15:00'  -->  '2023-03-05 18:30:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_START':
            # '2023-03-05 18:15:00'  -->  '2023-03-05 18:00:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta_half)

    # Data has END timestamp
    elif timestamp_index_name == 'TIMESTAMP_END':
        if new_timestamp_col == 'TIMESTAMP_END':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_MIDDLE':
            # '2023-03-05 18:30:00'  -->  '2023-03-05 18:15:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_START':
            # '2023-03-05 18:30:00'  -->  '2023-03-05 18:00:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta)

    # Data has START timestamp
    elif timestamp_index_name == 'TIMESTAMP_START':
        if new_timestamp_col == 'TIMESTAMP_START':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_MIDDLE':
            # '2023-03-05 18:00:00'  -->  '2023-03-05 18:15:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_END':
            # '2023-03-05 18:00:00'  -->  '2023-03-05 18:30:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta)

    # Make new timestamp column the first column in data
    if insert_as_first_col:
        first_col = data.pop(new_timestamp_col)
        data.insert(0, new_timestamp_col, first_col)

    if verbose:
        detail(f"++Added new timestamp column {new_timestamp_col}: "
               f"first={data[new_timestamp_col].iloc[0]}, last={data[new_timestamp_col].iloc[-1]}",
               verbose=verbose)
        detail(f"Timestamp index unchanged: first={data.index[0]}, last={data.index[-1]}",
               verbose=verbose)

    if set_as_index:
        # This loses freq/freqstr info for timestamp index
        data = data.set_index(new_timestamp_col)

        # Make sure timestamp index has freq/freqstr info
        data = TimestampSanitizer(data=data,
                                  output_middle_timestamp=False,
                                  nominal_freq=timestamp_freqstr).get()

    return data


def convert_series_timestamp_to_middle(data: Union[Series, DataFrame], verbose: bool = False) -> Union[
    Series, DataFrame]:
    """
    Convert timestamp index to show middle of averaging period.

    Adjusts the timestamp index to represent the MIDDLE of the averaging period,
    which simplifies data aggregation and plotting. If timestamps already represent
    the middle, no conversion is performed.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index. Index name must be one of 'TIMESTAMP_END',
        'TIMESTAMP_START', or 'TIMESTAMP_MIDDLE'.
    verbose : bool, optional
        Print status message. Default is False.

    Returns
    -------
    Union[Series, DataFrame]
        Data with index converted to TIMESTAMP_MIDDLE format.

    Raises
    ------
    ValueError
        If timestamp index name is not one of the recognized conventions.

    Notes
    -----
    **Why middle-of-period timestamps?**

    Using TIMESTAMP_END can lead to incorrect aggregation windows. For example,
    half-hourly data on 2022-07-26 with END timestamps starts at 2022-07-26 00:30
    and ends at 2022-07-27 00:00. When aggregated by date, this includes data
    outside the calendar day, leading to misaligned first and last records.

    With TIMESTAMP_MIDDLE, the first timestamp is 2022-07-26 00:15 and the last is
    2022-07-26 23:45, which correctly represent the full calendar day.

    **Timestamp conversions** (assuming 30-minute period):

    - TIMESTAMP_END 12:00   → TIMESTAMP_MIDDLE 11:45
    - TIMESTAMP_START 12:00 → TIMESTAMP_MIDDLE 12:15
    - TIMESTAMP_MIDDLE 12:15 → (no change)

    Examples
    --------
    >>> df.index.name = 'TIMESTAMP_END'
    >>> df_middle = convert_series_timestamp_to_middle(df, verbose=True)
    >>> print(df_middle.index.name)
    TIMESTAMP_MIDDLE
    """
    timestamp_name_before = data.index.name
    timestamp_name_after = 'TIMESTAMP_MIDDLE'

    if timestamp_name_before == timestamp_name_after:
        return data

    timestamp_freq = data.index.freq

    first_timestamp_before = data.index[0]
    last_timestamp_before = data.index[-1]

    if timestamp_name_before == 'TIMESTAMP_MIDDLE':
        if verbose:
            info("Convert to middle-of-period: OK (already middle)", verbose=verbose)
    else:
        timedelta = pd.to_timedelta(timestamp_freq) / 2
        if timestamp_name_before == 'TIMESTAMP_END':
            data.index = data.index - pd.Timedelta(timedelta)
        elif timestamp_name_before == 'TIMESTAMP_START':
            data.index = data.index + pd.Timedelta(timedelta)
        else:
            raise ValueError(
                f"Cannot convert timestamp format: index name '{timestamp_name_before}' "
                f"is not recognized. Expected one of 'TIMESTAMP_END', 'TIMESTAMP_START', "
                f"or 'TIMESTAMP_MIDDLE'. This indicates the time period the timestamp "
                f"represents. Check your data or rename the index before processing."
            )
        if verbose:
            info("Convert to middle-of-period: OK", verbose=verbose)

    data.index.name = 'TIMESTAMP_MIDDLE'

    return data


def add_timezone_info(timestamp_index, timezone_of_timestamp: str):
    """Add timezone info to timestamp index

    No data are changed, only the timezone info is added to the timestamp.

    :param: timezone_of_timestamp: If 'None', no timezone info is added. Otherwise
        can be `str` that describes the timezone in relation to UTC in the format:
        'UTC+01:00' (for CET), 'UTC+02:00' (for CEST), ...
        InfluxDB uses this info to upload data (always) in UTC/GMT.

    see: https://www.atmos.albany.edu/facstaff/ktyle/atm533/core/week5/04_Pandas_DateTime.html#note-that-the-timezone-is-missing-the-read-csv-method-does-not-provide-a-means-to-specify-the-timezone-we-can-take-care-of-that-though-with-the-tz-localize-method

    """
    return timestamp_index.tz_localize(timezone_of_timestamp)  # v0.3.1


def remove_after_date(data: Union[Series, DataFrame], yearly_end_date: str) -> Union[Series, DataFrame]:
    """
    Remove data after specifified date

    Args:
        data: Data with timestamp index
        yearly_end_date: Month and day after which all data will be removed
            Example:
                "08-11" means that all data after 11 August will be removed,
                this is done for each year in the dataset. For a dataset that
                contains data of multiple years, e.g. 2016, 2017 and 2018, the
                returned dataset will only contain data from all years, but
                data for each year will end on 1 August (i.e., 11 August 2016,
                11 August 2017 and 11 August 2018).

    Returns:
        Data with all data after *yearly_end_date* removed
    """
    month = int(yearly_end_date[0:2])
    dayinmonth = int(yearly_end_date[3:])
    data.loc[(data.index.month > month)] = np.nan
    data.loc[(data.index.month == month) & (data.index.day > dayinmonth)] = np.nan
    data = data.dropna()
    return data


def keep_years(data: Union[Series, DataFrame],
               start_year: int = None,
               end_year: int = None) -> Union[Series, DataFrame]:
    """
    Keep data between start and end year

    Args:
        data: Data with timeseries index
        start_year: First year of kept data
        end_year: Last year of kept data

    Returns:
        Data between start year and end year
    """
    if start_year:
        data = data.loc[data.index.year >= start_year]
    if end_year:
        data = data.loc[data.index.year <= end_year]
    return data


def keep_daterange(data: Union[Series, DataFrame],
                   start=None,
                   end=None,
                   verbose: bool = False) -> Union[Series, DataFrame]:
    """Keep only records whose timestamp falls within ``[start, end]`` (inclusive).

    A non-destructive date-range subselection: returns a copy restricted to the
    given window, leaving the input untouched so the caller can keep the full
    record and revert. Either bound may be omitted to leave that side open.

    Args:
        data: Series or DataFrame with a ``DatetimeIndex``.
        start: Lower bound (inclusive). A ``pandas.Timestamp``, ``datetime``, or
            any string pandas can parse (e.g. ``'2021-06-01'``,
            ``'2021-06-01 12:30'``). ``None`` leaves the start open.
        end: Upper bound (inclusive), same accepted types as ``start``. ``None``
            leaves the end open.
        verbose: Print how many records were kept.

    Returns:
        A copy of ``data`` containing only the in-range records. With both bounds
        ``None`` the full data is returned (as a copy).

    Raises:
        TypeError: If ``data`` does not have a ``DatetimeIndex``.
        ValueError: If ``start`` is after ``end``.
    """
    if not isinstance(data.index, DatetimeIndex):
        raise TypeError("keep_daterange requires data with a DatetimeIndex.")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"start ({start_ts}) is after end ({end_ts}).")

    keep = np.ones(len(data), dtype=bool)
    if start_ts is not None:
        keep &= data.index >= start_ts
    if end_ts is not None:
        keep &= data.index <= end_ts

    out = data.loc[keep].copy()
    if verbose:
        info(f"Kept {len(out)} of {len(data)} records "
             f"in date range [{start_ts}, {end_ts}].")
    return out


def calc_doy_timefraction(input_series: Series) -> DataFrame:
    df = pd.DataFrame(input_series)
    df['YEAR'] = df.index.year
    df['DOY'] = df.index.dayofyear
    df['TIMEFRACTION'] = (df.index.hour
                          + (df.index.minute / 60)
                          + (df.index.second / 3600)) / 24
    df['DOY_TIME'] = df['DOY'].add(df['TIMEFRACTION'])
    df[input_series.index.name] = df.index
    return df


def doy_cumulatives_per_year(series: Series) -> DataFrame:
    df = calc_doy_timefraction(input_series=series)
    return df.pivot(index='DOY_TIME', columns='YEAR', values=series.name).cumsum()


def doy_mean_cumulative(cumulatives_per_year_df: DataFrame,
                        excl_years_from_reference: list = None) -> DataFrame:
    """
    Calculate mean cumulative values by day-of-year with confidence intervals.

    Computes mean, standard deviation, and 95% confidence intervals for cumulative
    values across multiple years, organized by day-of-year and intra-day time fraction.

    Parameters
    ----------
    cumulatives_per_year_df : DataFrame
        DataFrame with day-of-year time (DOY_TIME) as index and years as columns,
        containing cumulative values for each year.
    excl_years_from_reference : list, optional
        Years to exclude from reference statistics. Default is None (use all years).

    Returns
    -------
    DataFrame
        Statistics by day-of-year with columns:
        - MEAN_DOY_TIME: Mean cumulative value
        - SD_DOY_TIME: Standard deviation
        - MEAN+SD, MEAN-SD: ±1 SD confidence bounds
        - MEAN+1.96_SD, MEAN-1.96_SD: ±1.96 SD (95%) confidence bounds
    """
    reference_years_df = cumulatives_per_year_df.copy()
    if excl_years_from_reference:
        for yr in excl_years_from_reference:
            try:
                reference_years_df = reference_years_df.drop(yr, axis=1)
            except KeyError:
                pass
    df = pd.DataFrame()
    df['MEAN_DOY_TIME'] = reference_years_df.mean(axis=1)
    df['SD_DOY_TIME'] = reference_years_df.std(axis=1)
    df['MEAN+SD'] = df['MEAN_DOY_TIME'].add(df['SD_DOY_TIME'])
    df['MEAN-SD'] = df['MEAN_DOY_TIME'].sub(df['SD_DOY_TIME'])
    df['1.96_SD_DOY_TIME'] = df['SD_DOY_TIME'].multiply(1.96)
    df['MEAN+1.96_SD'] = df['MEAN_DOY_TIME'].add(df['1.96_SD_DOY_TIME'])
    df['MEAN-1.96_SD'] = df['MEAN_DOY_TIME'].sub(df['1.96_SD_DOY_TIME'])
    return df


def calc_true_resolution(num_records: int,
                         data_nominal_res: float,
                         expected_records: int,
                         expected_duration: int):
    """
    Calculate the true resolution of the raw data

    Parameters
    ----------
    num_records: Number of raw data records.
    data_nominal_res: Nominal time resolution of the raw data, e.g. 0.05 for 20 Hz data
        (one measurement every 0.05 seconds)
    expected_records: Expected number of records in the raw data file, based on the nominal
        time resolution of the raw data.
    expected_duration: Expected duration of the raw data file in seconds.

    Returns
    -------
    float that gives the true resolution in seconds, e.g. 0.05s for 20 Hz (1/20 = 0.05)
    """
    ratio = num_records / expected_records
    if (ratio > 0.999) and (ratio < 1.001):
        # file_complete = True
        true_resolution = np.float64(expected_duration / num_records)
    else:
        # file_complete = False
        true_resolution = data_nominal_res
    return true_resolution


def create_timestamp(df, file_start, data_nominal_res, expected_duration):
    """Calculate the timestamp for each record in a dataframe.

    Insert true timestamp based on number of records in the file and the
    file duration.

    Files measured at a given time resolution may still produce
    more or less than the expected number of records.

    For example, a six-hour file with data recorded at 20Hz is expected to have
    432 000 records, but may in reality produce slightly more or less than that
    due to small inaccuracies in the measurements instrument's internal clock.
    This in turn would mean that the defined time resolution of 20Hz is not
    completely accurate with the true frequency being slightly higher or lower.

    This causes a (minor) issue when merging mutliple data files due to overlapping
    record timestamps, i.e. the last timestamp in file #1 is the same as the first
    timestamp in file #2, resulting in duplicate entries in the timestamp index column
    during merging of files #1 and #2.

    In addition, sometimes more than one timestamp can overlap, resulting in more
    overlapping timestamps and therefore more data loss. Although this data loss is
    minor (e.g. 3 records per 432 000 records), missing records are not desirable when
    calculating covariances between times series. The time series must be as complete
    and without missing records as possible to avoid errors.

    Args:
        df: Raw data without timestamp. In case data already have a timestamp, it
            will be overwritten.
        file_start: Start time of the file.
        data_nominal_res: Nominal time resolution of the raw data, e.g. 0.05 (one measurement every
            0.05 seconds, 20 Hz).
        expected_duration: Expected duration of the raw data file in seconds, e.g. 1800 for a
            30-minute file.

    Returns:
        df: pandas DataFrame with timestamp index
        true_resolution: time resolution of raw data measurements

    """
    n_records = len(df)
    expected_records = int(expected_duration / data_nominal_res, )
    true_resolution = calc_true_resolution(num_records=n_records, data_nominal_res=data_nominal_res,
                                           expected_records=expected_records, expected_duration=expected_duration)
    df['sec'] = df.index * true_resolution
    df['file_start_dt'] = file_start
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt']) \
                      + pd.to_timedelta(df['sec'], unit='s')
    df = df.drop(['sec', 'file_start_dt'], axis=1, inplace=False)
    df = df.set_index('TIMESTAMP', inplace=False)
    df.index = df.index.round(freq='50ms')  # Round to 50 ms accuracy
    return df, true_resolution


if __name__ == '__main__':
    pass
