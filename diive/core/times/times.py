"""
TIMES: TIMESTAMPS
=================

Timestamp sanitization, frequency detection and timestamp conversions.

Part of the diive library: https://github.com/holukas/diive
"""
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
    9. Make timestamp continuous without date gaps (records that are not on
       the time grid are removed and counted)
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
        If the time resolution cannot be detected (see :class:`DetectFrequency`).

    Warnings
    --------
    - **Regularization side effect**: When regularize=True, gaps are filled with
      NaN data rows. This may increase the number of rows significantly. The
      grid follows the position within the time step that most records share,
      so one stray or late record does not move it. Records that are not on
      the grid (for example 1-min records in 10-min data) are removed. Their
      number is shown as a warning and reported as 'rows_removed_off_grid' in
      :meth:`get_status`. The warning is shown also with verbose=False, unless
      the global verbosity is silent.
    - **Processing runs in the constructor**: if a step raises, no sanitized
      data are returned. The input data are not changed.
    - **Frequency detection**: Detection raises RuntimeError if timestamps are
      too irregular. nominal_freq=None skips only the comparison with the
      expected frequency, not the detection itself.

    See Also
    --------
    examples/times/times_timestamp_sanitizer.py : Examples with clean data,
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
            Generate continuous timestamp without date gaps. Records that are
            not on the time grid are removed and counted. Default is True.
        nominal_freq : str, optional
            Expected time resolution of data timestamp index. If provided, detected
            frequency is validated against this. Raises ValueError if they don't match.
            Examples: '10s', '5s', 's', '30min', '5min', 'min', '1h', '3h', 'h'.
            Default is None (no frequency validation).
        verbose : bool, optional
            Print status messages during processing. Default is False.

        Examples
        --------
        >>> import diive as dv, pandas as pd
        >>> idx = pd.date_range('2024-06-01 00:30', periods=48, freq='30min', name='TIMESTAMP_END')
        >>> df = pd.DataFrame({'TA': range(48)}, index=idx)
        >>> clean = dv.times.TimestampSanitizer(data=df, nominal_freq='30min').get()

        See ``examples/times/times_timestamp_sanitizer.py`` for clean data, minor
        issues, badly broken timestamps, and selective processing.
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
        self._off_grid_removed = 0
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
        """Return the sanitized DataFrame."""
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
            - 'rows_removed_off_grid': Rows removed by regularization because
              their timestamp is not on the time grid
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
        >>> import diive as dv, pandas as pd
        >>> idx = pd.date_range('2024-06-01 00:30', periods=48, freq='30min', name='TIMESTAMP_END')
        >>> sanitizer = dv.times.TimestampSanitizer(pd.DataFrame({'TA': range(48)}, index=idx))
        >>> status = sanitizer.get_status()  # rows_removed, frequency_confidence, ...
        """
        return {
            'original_shape': self._original_shape,
            'final_shape': self.data.shape,
            'rows_removed': self._nat_removed + self._duplicates_removed + self._off_grid_removed,
            'rows_removed_nat': self._nat_removed,
            'rows_removed_duplicates': self._duplicates_removed,
            'rows_removed_off_grid': self._off_grid_removed,
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
            self.data, self._off_grid_removed = _regularize(data=self.data, freq=self.inferred_freq,
                                                            verbose=self.verbose)
            self._rows_added_by_regularization = len(self.data) - (rows_before - self._off_grid_removed)

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

    Two regular timestamps are enough, since they define one interval. An index
    with a single timestamp defines no interval at all and therefore raises
    RuntimeError, same as an index that is too irregular to detect.

    Three methods run: pandas ``infer_freq`` on the full index, the most
    frequent time step, and ``infer_freq`` on chunks at the start and end of the
    index. After construction, ``freq`` (also returned by :meth:`get`),
    ``confidence`` (0-1), ``detection_method``, ``percent_matching`` and
    ``alternatives`` describe the result.

    Args:
        index: Timestamp index to check.
        verbose: Print the detected frequency.

    - Example: examples/times/times_frequency_detection.py
    - Tests: tests/test_time.py

    """

    def __init__(self, index: pd.DatetimeIndex, verbose: bool = False):
        """Set up frequency detection for an index. See the class docstring."""
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
        # The private variant also hands over the exact match fraction. Parsing it back
        # out of freqinfo_timedelta rounded it to whole percent, so a genuine 99.9%
        # looked like a perfectly regular record.
        freq_timedelta, freqinfo_timedelta, match_timedelta = _infer_freq_from_timedelta(timestamp_ix=self.index)
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
                # % matching comes from the timedelta method, the only one that measures
                # it. All methods agreeing means that method found a frequency, so the
                # fraction is always available here.
                self.percent_matching = match_timedelta * 100
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
            # Freq found from the most frequent time step, which occurs in
            # more than 50% of all intervals
            self.freq = freq_timedelta
            self.detection_method = "timedelta"
            # This branch runs only if the timedelta method found a frequency, which is
            # exactly when it also reports its match fraction.
            self.confidence = match_timedelta
            self.percent_matching = match_timedelta * 100
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
                "Could not detect timestamp frequency with any method (pandas "
                "inferred_freq, most-frequent timedelta, start/end chunks). The "
                "timestamps are too irregular: no single spacing occurs often "
                "enough to identify. To fix, make the index regular before "
                "calling, e.g. resample or reindex it onto the intended "
                "frequency. Detection cannot be skipped: TimestampSanitizer's "
                "nominal_freq only validates an already-detected frequency, and "
                "regularize runs after detection, so neither bypasses this."
            )

    def get(self) -> str:
        """Return the detected frequency string."""
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


# A run of at least this many records with the same resolution is a resolution
# period. Random missing records almost never leave this many records in a row
# at a multiple of the resolution: at 50 % missing 10MIN records the chance per
# record is 0.25**12 (6e-8) for 20MIN spacing and lower for coarser ones.
_MIN_PERIOD_RECORDS = 12
# A shorter run is still a period if it has at least this many records and missing
# records of the neighbouring periods are as unlikely to produce it (see
# _min_period_records).
_MIN_SHORT_PERIOD_RECORDS = 3


def _min_period_records(m: np.ndarray) -> np.ndarray:
    """Records a run needs to be a period when its spacing is *m* times the
    resolution of a neighbouring period.

    Missing records of resolution r give spacing m*r with a chance of at most
    q = (m-1)**(m-1) / m**m per record (at the worst missing fraction (m-1)/m).
    The run needs as many records as make q**n as small as 0.25**12, the chance
    behind _MIN_PERIOD_RECORDS at m = 2: 12 records at m = 2, 9 at 3, 6 at 10,
    4 at 30 and 60, never fewer than 3.
    """
    m = np.asarray(m, dtype=float)
    log_q = (m - 1) * np.log(m - 1) - m * np.log(m)
    n = _MIN_PERIOD_RECORDS * np.log(0.25) / log_q
    return np.maximum(_MIN_SHORT_PERIOD_RECORDS, np.ceil(n - 1e-9)).astype(int)


# Short blocks need a spacing of at least this many times the finer spacing
# around them. Below it, regular losses such as every other record missing are
# common and the general rule (_min_period_records) applies.
_MIN_BLOCK_RATIO = 5


def _min_block_records(m: np.ndarray, context: np.ndarray = 5) -> np.ndarray:
    """Spacings a block needs to be a period when its spacing is *m* times the
    spacing of the finer records around it and it sits on its own grid.

    The block shows k spacings of m*r, with *context* other records present
    around them (see _coarse_blocks): 5 between finer records (three before,
    two after for TIMESTAMP_END). Random missing records of resolution r,
    missing fraction p, leave this pattern at a given record with a chance of
    (1-p)**(k+context) * p**(k*(m-1)): k+context records present and the m-1
    records before each block record missing. A gap costs nothing (it may be a
    reboot). Only 1 in m records lies on the grid of the coarse spacing. The
    block needs as many spacings as make the worst case over p as small as
    0.25**12 (see _MIN_PERIOD_RECORDS). With context 5: 4 at m = 5, 3 at 6 to 9,
    2 at 10 and above. With context 3 (start or end of the data): 5 at m = 5,
    4 at 6 to 8, 3 at 9 to 14, 2 at 15 and above. A single spacing is never
    enough: one outage of exactly m-1 records that ends on the grid explains it.
    """
    m = np.asarray(m, dtype=float)
    context = np.broadcast_to(np.asarray(context, dtype=float), m.shape)
    need = np.full(m.shape, _MIN_PERIOD_RECORDS)
    for k in range(_MIN_PERIOD_RECORDS - 1, 1, -1):
        present, missing = k + context, k * (m - 1)
        p = missing / (missing + present)
        log_chance = present * np.log(1 - p) + missing * np.log(p) - np.log(m)
        need = np.where(log_chance <= _MIN_PERIOD_RECORDS * np.log(0.25), k, need)
    return need


def _coarse_blocks(own: np.ndarray, wall_ns: np.ndarray) -> np.ndarray:
    """Short blocks of coarse records between finer records (L286, L302).

    *own* is the spacing to the previous record in ns (0 for the first record),
    *wall_ns* the local clock time in ns. For TIMESTAMP_START, pass both
    reversed, so that *own* is again the spacing that covers each record's
    own averaging period. A block is a run of k records with the same own
    spacing g, on the grid of g (e.g. 10MIN records at :00, :10, ...), g a
    whole multiple m >= _MIN_BLOCK_RATIO of a finer spacing r on its sides.

    Before the run: two records at spacing r, then optionally one record after
    a gap (own spacing longer than g, a whole multiple of r), which joins the
    block. Or the run starts at the second record: the first record has no own
    spacing and joins the block. After the run: two records at spacing r,
    optionally after a gap (a first record at any whole multiple of r), or the
    end of the data. One side must have finer records. The run needs
    _min_block_records(m, context) records, context being the other records
    the pattern needs: 3 before the run (plus the record after a gap) or 1 for
    the first record, 2 after it (3 after a gap) or 0 at the end of the data.
    Returns the block resolution of each record in ns, 0 outside blocks.
    """
    n = len(own)
    starts = np.flatnonzero(np.concatenate([[True], own[1:] != own[:-1]]))
    lengths = np.diff(np.concatenate([starts, [n]]))
    ends = starts + lengths - 1
    g = own[starts]

    def at(i):
        # Own spacing at positions i, 0 outside the data
        return np.where((i >= 0) & (i < n), own[np.clip(i, 0, n - 1)], 0)

    # Before the run
    edge_l = starts == 1
    gap_l = at(starts - 1) > g
    first = np.where(gap_l | edge_l, starts - 1, starts)
    r_l = at(first - 1)
    ctx_l = ~edge_l & (r_l > 0) & (at(first - 2) == r_l)
    ctx_l &= ~gap_l | (at(first) % np.where(r_l > 0, r_l, 1) == 0)
    # After the run
    edge_r = ends == n - 1
    r_n, r_g = at(ends + 1), at(ends + 2)
    norm_r = ~edge_r & (r_n > 0) & (r_g == r_n)
    gap_r = ~edge_r & ~norm_r & (r_g > 0) & (at(ends + 3) == r_g) & (r_n % np.where(r_g > 0, r_g, 1) == 0)
    r_r = np.where(norm_r, r_n, np.where(gap_r, r_g, 0))

    ok = (ctx_l | edge_l) & (norm_r | gap_r | edge_r) & (ctx_l | norm_r | gap_r)
    ok &= ~(ctx_l & (norm_r | gap_r)) | (r_l == r_r)
    r = np.where(ctx_l, r_l, r_r)
    safe_r = np.where(ok & (r > 0), r, 1)
    ok &= (r > 0) & (g >= _MIN_BLOCK_RATIO * safe_r) & (g % safe_r == 0)
    ok &= wall_ns[starts] % np.where(ok, g, 1) == 0
    context = np.where(edge_l, 1, 3 + gap_l) + np.where(norm_r, 2, np.where(gap_r, 3, 0))
    ok[ok] = lengths[ok] >= _min_block_records(g[ok] // r[ok], context[ok])
    block = np.repeat(np.where(ok, g, 0), lengths)
    block[first[ok]] = g[ok]
    return block


def detect_freq_groups(index: DatetimeIndex) -> Series:
    """
    Assign each record to the time resolution (frequency group) it was recorded at

    Used for data whose time resolution changes over time, e.g. a sensor that logged
    10MIN averages for a year and 1MIN averages afterwards. The result can be used to
    access the different time resolution groups separately during later processing.

    The assignment works in four passes:

    1. A record whose time differences to the previous and to the next record are
       equal has an unambiguous resolution: that difference.
    2. A record where the two differences differ sits next to a timestamp gap or at a
       change of resolution. It gets the group of a neighbour from pass 1 (the
       difference to that neighbour equals the neighbour's resolution). If both
       neighbours are from pass 1, the side of the record's own averaging period
       wins: the previous record for TIMESTAMP_END (and unnamed) indexes, the next
       record for TIMESTAMP_START. Two exceptions: a neighbour that is a single
       pass-1 record, with no neighbour of the same group, loses if the other
       neighbour belongs to a run of pass-1 records and the single record's
       resolution is a whole multiple of the run's. It may be one record between
       missing records of that resolution (see pass 3). It also loses if its
       resolution is half the run's and the run goes on beyond it: a stray
       stamp at 12:30:30 in 1MIN data does not pull the record at 12:31 into
       its 30 s group.
    3. A single record between two gaps of the same length (e.g. 10MIN data with the
       records before and after it missing) has equal differences to both neighbours
       and got the gap length in pass 1. It joins its neighbours' group when both have
       the same resolution and the gaps are whole missing records of it.
    4. Many missing records leave short runs that look like a coarser resolution
       (10MIN data with every other record missing for an hour give 20MIN spacing)
       and records that fit no neighbour. Regular outages, such as 2 of every 4
       records missing (1, 3, 1, 3 steps) or every 3rd and every 4th (1, 3,
       2, 3, 1, 2 steps), leave spacings that passes 1-3 cannot assign. Such
       a record gets the greatest common divisor g of its two spacings if it
       lies in a stretch of at least 24 spacings that are all whole multiples
       of g, and either its shorter spacing is g and covers its own averaging
       period, or both its spacings are at most 6 times g and the stretch
       holds at least 12 such records. Their neighbours with spacings of at
       most 6 times g get g too. A stretch that shows only multiples of a
       coarser step gets that step: spacings of 4 and 6 minutes give 120 s,
       also in 1MIN data (see ``_regular_outages``). Unassigned records
       between two such records count towards their run. A run is a series
       of consecutive records with the same resolution and only whole
       multiples of it between them.
       Unassigned records count towards the run they continue if one of their
       spacings equals its resolution (e.g. the records after a missing record
       at the end of a 30MIN run). A run of at least 12
       records is a resolution period. A run of 3 to 11 records is a
       period too unless missing records of the nearest such period could produce
       it: its resolution must not be a whole multiple m of that period's, or the
       run needs more records the smaller m is (12 at m = 2, 9 at 3, 6 at 10, 4 at
       30 and 60). Eleven 10MIN records between 1MIN periods stay 600 s, while
       10MIN data with every other record missing for an hour join the 10MIN
       period. A block of evenly spaced records between finer records needs
       fewer records if it sits on its own grid (10MIN records at :00, :10, ...),
       each record covers a whole period of the block's resolution, and at least
       two records at one finer spacing follow on each side: 2 records at 10 or
       more times the finer spacing, 3 at 6 to 9, 4 at 5 (see
       ``_min_block_records``). Two 10MIN records between 1MIN periods get
       600 s. A single coarse record stays ambiguous: one outage in the finer
       data can leave it, so it gets the finer resolution. A gap (e.g. a
       reboot) may separate the block from the finer records. A block record
       right after a gap counts only as context, because the gap hides its
       spacing: 10MIN records between 1MIN data need 3 records with a gap
       before the block (TIMESTAMP_END), 2 with a gap after it. At the start or
       end of the data one side has no records, and the block needs more
       records: at 10 times the finer spacing 4 at the start, 3 at the end
       for TIMESTAMP_END, because the first record has no spacing of its own
       (reversed for TIMESTAMP_START). Blocks apply also with fewer than 12
       records.
       The last coarse record before finer data keeps the resolution G of the
       period before it (on its own side) if its own spacing is a whole
       multiple of G, at most 6 times G, and on the period's grid: the last
       30MIN record before 1MIN data stays 1800 s when the record before it
       is missing. The first 1MIN record after a short gap that ends on the
       30MIN grid gets 1800 s as well. Likewise, the first record of a coarse
       run gets the finer resolution of the period before it if its own
       spacing is one step of that period (see ``_keep_coarse_edge``).
       Every other record joins the nearest period on either side if all
       time differences between it and that period, and to both its own
       neighbours, are whole multiples of the period's resolution. If periods of
       different resolutions qualify, the coarsest wins: a record spaced at
       multiples of 10MIN shows no sign of 1MIN logging. Without any run of 12
       records (e.g. 90 % of the records missing at random) pass 4 changes
       nothing but the short blocks and the regular outages.

    Example: in regular 10MIN data with the record at 12:20 missing, the records at
    12:10 and 12:30 have differences of 10 and 20 minutes to their neighbours. Both
    are assigned to the 600 s group from their regular neighbour, so a single missing
    record does not remove the records around it.

    Example: 1MIN data up to 10:00 (TIMESTAMP_END), then 10MIN data from 10:10. The
    record at 10:00 covers 09:59-10:00 and is assigned to the 60 s group, the record
    at 10:10 to the 600 s group.

    Note:
        Sometimes there are transition periods between one time resolution and another.
        For example, when the time resolution changes from 10MIN to 1MIN, there might be
        several records in between that have neither 10MIN nor 1MIN, but e.g. 7S or 29MIN etc.
        For these transitional records, the time resolution is not clear and they stay
        unassigned (NaN). This typically affects only a handful of records during the
        transition period(s).

    Args:
        index: Sorted timestamp index.

    Returns:
        Series 'FREQ_AUTO_SEC' on *index* with the time resolution of each record in
        seconds, NaN where the resolution could not be determined. All NaN for
        fewer than two records.

    :: Added in v0.43.0
    """
    if len(index) < 2:
        # Callers raise their own error for too few records
        return pd.Series(index=index, data=np.nan, name='FREQ_AUTO_SEC', dtype=float)
    # Work in integer nanoseconds so the multiple-of check is exact also for
    # sub-second resolutions. A difference of 0 marks "no neighbour on this side".
    ns = index.as_unit('ns').asi8
    diffs = np.diff(ns)
    prev_d = np.concatenate([[0], diffs])
    next_d = np.concatenate([diffs, [0]])

    # Pass 1: equal spacing on both sides
    group = np.where((prev_d == next_d) & (prev_d > 0), prev_d, 0)

    # Pass 2: records next to a gap or a change of resolution. Neighbour groups
    # are taken from pass 1 only, so the result does not depend on the order in
    # which ambiguous records are visited.
    prev_group = np.concatenate([[0], group[:-1]])
    next_group = np.concatenate([group[1:], [0]])

    # A pass-1 record is backed by a run if a neighbour has the same group. A lone
    # record between two equal gaps is not. If its group is a whole multiple of a
    # run on the other side, it may be a single record between missing records of
    # that resolution (pass 3), and it must not pull the record into its
    # gap-sized group.
    in_run = (group > 0) & ((group == prev_group) | (group == next_group))
    prev_in_run = np.concatenate([[False], in_run[:-1]])
    next_in_run = np.concatenate([in_run[1:], [False]])

    # Pass-1 group three records away: beyond a lone neighbour and the record
    # on its far side
    prev_far = np.concatenate([np.zeros(3, dtype=group.dtype), group[:-3]])[:len(group)]
    next_far = np.concatenate([group[3:], np.zeros(3, dtype=group.dtype)])[-len(group):]

    def _score(d, g, g_in_run, g_other, other_in_run, far):
        # 2 = spacing equals the neighbour's resolution, 1 = the same, but the
        # neighbour is a lone record that may sit between missing records of the
        # run on the other side, or a stray stamp halfway between two records of
        # that run, with the run going on beyond them (L304), 0 = does not fit.
        # A pass-1 neighbour's group always equals the spacing to it, so whole
        # multiples do not occur here.
        valid = (d > 0) & (g > 0)
        exact = valid & (d == g)
        safe_o = np.where(other_in_run, g_other, 1)
        holes = (g > g_other) & (g % safe_o == 0)
        stray = (2 * g == g_other) & (far == g_other)
        suspect = ~g_in_run & other_in_run & (holes | stray)
        return np.where(exact, np.where(suspect, 1, 2), 0)

    prev_score = _score(prev_d, prev_group, prev_in_run, next_group, next_in_run, prev_far)
    next_score = _score(next_d, next_group, next_in_run, prev_group, prev_in_run, next_far)

    # On a tie, take the side covered by the record's own averaging period
    tie_to_next = index.name == 'TIMESTAMP_START'
    take_next = (next_score > prev_score) | ((next_score == prev_score) & tie_to_next)
    fallback = np.where(take_next, next_group, prev_group)
    fallback = np.where(np.maximum(prev_score, next_score) > 0, fallback, 0)

    group = np.where(group > 0, group, fallback)

    # Pass 3: lone pass-1 record between two equal gaps
    prev_final = np.concatenate([[0], group[:-1]])
    next_final = np.concatenate([group[1:], [0]])
    lone = (group > 0) & ~in_run & (prev_final > 0) & (prev_final == next_final)
    lone &= prev_d % np.where(lone, prev_final, 1) == 0
    group = np.where(lone, prev_final, group)

    own_d = next_d if tie_to_next else prev_d
    group, outage = _regular_outages(group, prev_d, next_d, own_d)

    wall = index.tz_localize(None) if index.tz is not None else index
    wall_ns = wall.as_unit('ns').asi8
    if tie_to_next:
        block = _coarse_blocks(next_d[::-1], wall_ns[::-1])[::-1]
    else:
        block = _coarse_blocks(prev_d, wall_ns)
    group = _join_resolution_periods(group, diffs, block, outage, tie_to_next)

    group_sec = np.where(group > 0, group / 1e9, np.nan)
    return pd.Series(index=index, data=group_sec, name='FREQ_AUTO_SEC')


def describe_time_resolution(index: DatetimeIndex) -> str:
    """Short text of the time resolution(s) of the records in *index*.

    Each record gets its resolution from ``detect_freq_groups``, so missing records
    and changes of resolution are handled. One resolution gives e.g. ``'30min'``.
    Several give each with its number of records, in order of first appearance,
    e.g. ``'10min (8,640 records), 1min (52,000 records)'``. Records without a
    detected resolution count as ``unknown``, and fewer than two records give
    ``'unknown'``.

    Example:
        >>> import pandas as pd
        >>> from diive.core.times.times import describe_time_resolution
        >>> describe_time_resolution(pd.date_range('2024-01-01', periods=48, freq='30min'))
        '30min'
    """
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        return 'unknown'
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    ns = index.as_unit('ns').asi8
    diffs = np.diff(ns)
    if diffs[0] > 0 and (diffs == diffs[0]).all():
        # Evenly spaced, nothing to detect (also skips detection on long regular data)
        sec = np.full(len(index), diffs[0] / 1e9)
    else:
        sec = detect_freq_groups(index).to_numpy()
    resolutions = pd.Series(sec).value_counts(dropna=False, sort=False)
    if len(resolutions) == 1 and resolutions.index[0] == resolutions.index[0]:  # one, not NaN
        return _resolution_str(resolutions.index[0])
    return ', '.join(f"{_resolution_str(s)} ({n:,} records)" for s, n in resolutions.items())


def _resolution_str(seconds: float) -> str:
    """Offset string of a resolution in seconds with the count, e.g. '1min', 'unknown' for NaN."""
    if seconds != seconds:
        return 'unknown'
    offset = to_offset(pd.Timedelta(int(round(seconds * 1e9)), unit='ns'))
    return f"{offset.n}{offset.name}"


# Spacings a stretch needs before its common resolution counts as evidence
# (_regular_outages). Random missing records leave a spacing that is a whole
# multiple of twice their resolution with a chance below 1/2, so 24 such
# spacings in a row are as unlikely as 0.25**12 (see _MIN_PERIOD_RECORDS).
_MIN_OUTAGE_SPACINGS = 2 * _MIN_PERIOD_RECORDS
# Longest spacing, in steps of g, of a record that gets g from the common
# divisor of its spacings (_regular_outages), and of the last coarse record
# before finer data (_keep_coarse_edge).
_MAX_OUTAGE_STEPS = 6


def _regular_outages(group: np.ndarray, prev_d: np.ndarray, next_d: np.ndarray,
                     own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """*group* with a resolution for unassigned records in regular outage
    patterns (L301, L309), and a mask of the records that got one.

    When 2 of every 4 or 3 of every 5 records are missing, the spacings
    alternate (1, 3, 1, 3 or 1, 4, 1, 4 steps) and passes 1-3 assign nothing.
    Two patterns at once (every 3rd and every 4th record missing: 1, 3, 2, 3,
    1, 2 steps) or kept records that are never adjacent (records 2 and 5 of
    every 5: 3, 2 steps) do not even show a spacing that divides the other.
    An unassigned record gets g, the greatest common divisor of its two
    spacings, if it lies in a stretch of at least _MIN_OUTAGE_SPACINGS
    consecutive spacings that are all whole multiples of g and one of these
    holds:

    - its shorter spacing is g, covers its own averaging period (*own*, as in
      pass 2) and divides the longer one (any multiple),
    - both spacings are at most _MAX_OUTAGE_STEPS times g, and the stretch
      holds at least _MIN_PERIOD_RECORDS such records with g as their common
      divisor.

    A record next to a record that got g in this way also gets g if its
    own spacings have a coarser common divisor, are whole multiples of g
    and at most _MAX_OUTAGE_STEPS times g, and the stretch holds those
    _MIN_PERIOD_RECORDS records (spacings of 4 and 6 steps between records
    with spacings of 1 and 4). Processed from fine to coarse g, so a coarser
    g that also qualifies wins: a stretch of spacings of 4 and 6 minutes
    gets 120 s. Every spacing is a whole multiple of the finest resolution
    in the data, so a long stretch alone does not rule out a finer g: the
    12 records keep a few stray stamps at 12:30:30 in 1MIN data from giving
    their neighbours 30 s. The cap keeps timestamps that jitter around a
    resolution (599, 601 s) from getting a 1 s resolution, and the own side
    keeps the last 15MIN record before 1MIN data at 900 s. In pass 4 these
    labels form runs as usual (see _fill_outage_gaps).
    """
    lo, hi = np.minimum(prev_d, next_d), np.maximum(prev_d, next_d)
    g_loc = np.gcd(prev_d, next_d)
    base = (group == 0) & (lo > 0) & (hi > lo)
    divides = base & (own == lo) & (hi % np.where(base, lo, 1) == 0)
    cand = divides | (base & (hi <= _MAX_OUTAGE_STEPS * g_loc))
    if not cand.any():
        return group, np.zeros(len(group), dtype=bool)
    diffs = next_d[:-1]
    result = group.copy()
    # Ascending, so a coarser resolution that also fits overwrites a finer one
    for g in np.unique(g_loc[cand]):
        bad = diffs % g != 0
        # Stretch number of each spacing: each bad spacing starts a new one
        stretch = np.cumsum(bad)
        size = np.bincount(stretch[~bad], minlength=stretch[-1] + 1)
        # Records with g as their own common divisor, counted per stretch
        own_g = cand & (g_loc == g)
        evidence = np.bincount(stretch[np.flatnonzero(own_g)], minlength=stretch[-1] + 1)
        # Both spacings of a record are good and in the same stretch
        long = np.zeros(len(group), dtype=bool)
        dense = np.zeros(len(group), dtype=bool)
        long[1:-1] = size[stretch[1:]] >= _MIN_OUTAGE_SPACINGS
        dense[1:-1] = evidence[stretch[1:]] >= _MIN_PERIOD_RECORDS
        own_g &= long & ((divides & (lo == g)) | dense)
        # A record with a coarser common divisor next to such a record
        near = np.concatenate([[False], own_g[:-1]]) | np.concatenate([own_g[1:], [False]])
        near &= base & long & dense & (g_loc % g == 0) & (hi <= _MAX_OUTAGE_STEPS * g)
        result[own_g | near] = g
    return result, result != group


def _join_resolution_periods(group: np.ndarray, diffs: np.ndarray, block: np.ndarray,
                             outage: np.ndarray, own_next: bool) -> np.ndarray:
    """Pass 4 of :func:`detect_freq_groups`: find the resolution periods, then let
    every other record join the coarsest neighbouring period it fits.

    *group* is the resolution of each record in ns (0 = unassigned), *diffs* the
    time differences between consecutive records in ns, *block* the resolution
    of short coarse blocks from _coarse_blocks (0 = none), *outage* the records
    labelled by _regular_outages. *own_next* is True if the next record's side
    covers each record's own averaging period (TIMESTAMP_START).
    """
    n = len(group)
    group = np.where(block > 0, block, group)
    outage = outage & (block == 0)
    if n < _MIN_PERIOD_RECORDS:
        return group
    # Runs of one resolution. A run also ends at a difference that is no whole
    # multiple of its resolution: pass 2 labels a record from one neighbour only,
    # so in 1MIN data with many holes a run labelled 120 s could carry on across
    # a 1MIN step. Unassigned records count towards the run they continue (L289),
    # but stay unassigned until they join a period below.
    counted = _fill_outage_gaps(_continue_runs(group, diffs), group, diffs, outage)
    safe_g = np.where(counted[:-1] > 0, counted[:-1], 1)
    joined = (counted[1:] == counted[:-1]) & (diffs % safe_g == 0)
    starts = np.flatnonzero(np.concatenate([[True], ~joined]))
    lengths = np.diff(np.concatenate([starts, [n]]))
    run_group = counted[starts]
    long_run = (run_group > 0) & (lengths >= _MIN_PERIOD_RECORDS)
    if not long_run.any():
        return group
    period = np.repeat(long_run | _short_periods(run_group, lengths, long_run), lengths)
    period = (period & (group > 0)) | (block > 0)
    prev_d = np.concatenate([[0], diffs])
    next_d = np.concatenate([diffs, [0]])
    group, period = _keep_coarse_edge(group, period, block, diffs, prev_d, next_d, own_next)
    if period.all():
        return group

    pos = np.arange(n)
    # Nearest period record before and after each record (-1 / n = none)
    prev_p = np.maximum.accumulate(np.where(period, pos, -1))
    next_p = np.minimum.accumulate(np.where(period, pos, n)[::-1])[::-1]

    result = group.copy()
    # Ascending, so a coarser resolution that also fits overwrites a finer one
    for res in np.unique(group[period]):
        # bad[k]: number of differences up to record k that are no whole multiple
        # of res. The stretch between records a < b fits if bad[a] == bad[b].
        bad = np.concatenate([[0], np.cumsum(diffs % res != 0)])
        own = (prev_d % res == 0) & (next_d % res == 0)
        fits = np.zeros(n, dtype=bool)
        for near in (prev_p, next_p):
            safe = np.clip(near, 0, n - 1)
            fits |= (near >= 0) & (near < n) & (group[safe] == res) & (bad[safe] == bad)
        result = np.where(~period & own & fits, res, result)
    return result


def _keep_coarse_edge(group: np.ndarray, period: np.ndarray, block: np.ndarray, diffs: np.ndarray,
                      prev_d: np.ndarray, next_d: np.ndarray, own_next: bool) -> tuple[np.ndarray, np.ndarray]:
    """*group* and *period* with the last coarse record before finer data kept
    at the coarse resolution, and the first coarse record after finer data
    that is one finer step away kept at the finer one (L310).

    With the record before it missing, the last 30MIN record before 1MIN data
    has no pass-1 neighbour on its own side, and pass 2 gives it the 60 s of
    the record after it. A record outside a short block gets the resolution
    G of the nearest period record on its own side (the previous record for
    TIMESTAMP_END, the next for TIMESTAMP_START) instead of a finer one or
    none if:

    - its own spacing is a whole multiple of G, at most _MAX_OUTAGE_STEPS
      times G (up to five missing records),
    - its other spacing is no whole multiple of G (finer data follow),
    - every spacing back to that period record is a whole multiple of G (it
      sits on the period's grid).

    It then counts as a record of that period. The first 1MIN record after a
    short gap that happens to end on the 30MIN grid gets 1800 s too: the
    spacing alone cannot tell the two apart. In the mirror case, a record
    that took a coarser resolution from its other side gets G if its own
    spacing equals G: in holey 1MIN data the record at 10:00 before 10MIN
    data from 10:10 stays 60 s although the record before it got no
    resolution in passes 1-3.
    """
    n = len(group)
    pos = np.arange(n)
    own_d, other_d = (next_d, prev_d) if own_next else (prev_d, next_d)
    if own_next:
        near = np.minimum.accumulate(np.where(period, pos, n)[::-1])[::-1]
        near = np.concatenate([near[1:], [n]])
    else:
        near = np.maximum.accumulate(np.where(period, pos, -1))
        near = np.concatenate([[-1], near[:-1]])
    has = (near >= 0) & (near < n)
    g_near = np.where(has, group[np.clip(near, 0, n - 1)], 0)
    safe = np.where(g_near > 0, g_near, 1)
    coarse = (g_near > group) & (own_d % safe == 0) & (other_d % safe != 0)
    coarse &= own_d <= _MAX_OUTAGE_STEPS * g_near
    # Mirror case: the first record of a coarse run took its resolution from
    # its other side, but its own spacing is one step of the finer period
    fine = (g_near < group) & (own_d == g_near)
    cand = has & (block == 0) & (own_d > 0) & (coarse | fine)
    if not cand.any():
        return group, period
    group, period = group.copy(), period.copy()
    for res in np.unique(g_near[cand]):
        bad = np.concatenate([[0], np.cumsum(diffs % res != 0)])
        idx = np.flatnonzero(cand & (g_near == res))
        idx = idx[bad[near[idx]] == bad[idx]]
        group[idx] = res
        period[idx] = True
    return group, period


def _fill_outage_gaps(counted: np.ndarray, group: np.ndarray, diffs: np.ndarray, outage: np.ndarray) -> np.ndarray:
    """*counted* with unassigned records between two records of one regular
    outage pattern set to its resolution, for counting run lengths (L309).

    With 7 of every 10 records missing, records with spacings of 4 and 11
    steps get no resolution of their own (see _regular_outages) and split
    the pattern into runs of 2 to 4 records. An unassigned record counts
    with the pattern when the nearest assigned records on both sides got
    the same resolution g from _regular_outages and every spacing between
    them is a whole multiple of g.
    """
    n = len(group)
    pos = np.arange(n)
    has = group > 0
    prev_l = np.maximum.accumulate(np.where(has, pos, -1))
    next_l = np.minimum.accumulate(np.where(has, pos, n)[::-1])[::-1]
    cand = (counted == 0) & (prev_l >= 0) & (next_l < n)
    p, q = np.clip(prev_l, 0, n - 1), np.clip(next_l, 0, n - 1)
    cand &= outage[p] & outage[q] & (group[p] == group[q])
    if not cand.any():
        return counted
    counted = counted.copy()
    for g in np.unique(group[p[cand]]):
        bad = np.concatenate([[0], np.cumsum(diffs % g != 0)])
        idx = np.flatnonzero(cand & (group[p] == g))
        idx = idx[bad[p[idx]] == bad[q[idx]]]
        counted[idx] = g
    return counted


def _continue_runs(group: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    """*group* with each unassigned record set to the resolution of the run it
    continues, for counting run lengths.

    An unassigned record continues the run before it if all time differences
    from the run's last record to it are whole multiples of the run's
    resolution and it, like every unassigned record in between, is exactly one
    resolution away from a neighbour (the evidence pass 2 needs for a label).
    Otherwise it may continue the run after it under the same conditions. A
    30MIN run with one record missing near the end of the data leaves the last
    records without a group in passes 1-3. Counted with the run, they keep it
    long enough to be a period. Without the exact spacing, unassigned records
    in 1MIN data with half the records missing joined 2MIN runs into spurious
    periods.
    """
    n = len(group)
    pos = np.arange(n)
    zero = group == 0
    filled = group.copy()
    prev_d = np.concatenate([[0], diffs])
    next_d = np.concatenate([diffs, [0]])
    for order in (slice(None), slice(None, None, -1)):
        z, g_ = zero[order], group[order]
        dd = (prev_d if order.step is None else next_d)[order]
        last = np.maximum.accumulate(np.where(z, -1, pos))
        has = last >= 0
        g_last = np.where(has, g_[np.clip(last, 0, n - 1)], 0)
        safe = np.where(g_last > 0, g_last, 1)
        exact = (prev_d[order] == g_last) | (next_d[order] == g_last)
        bad = z & ((g_last == 0) | (dd % safe != 0) | ~exact)
        cum = np.cumsum(bad)
        fits = z & has & (cum == cum[np.clip(last, 0, n - 1)])
        f = filled[order]
        filled[order] = np.where(fits & (f == 0), g_last, f)
    return filled


def _short_periods(run_group: np.ndarray, lengths: np.ndarray, long_run: np.ndarray) -> np.ndarray:
    """Runs shorter than _MIN_PERIOD_RECORDS that are resolution periods too.

    A run of at least _MIN_SHORT_PERIOD_RECORDS records is a period unless missing
    records of the nearest long period on either side could explain it. They can
    only if the run's resolution is a whole multiple m of that period's, and then
    the run needs _min_period_records(m) records. Eleven 10MIN records between
    1MIN periods (m = 10, 6 needed) stay 600 s, three 20MIN records in 10MIN data
    (m = 2, 12 needed) join the 10MIN period.
    """
    nr = len(run_group)
    idx = np.arange(nr)
    need = np.full(nr, _MIN_SHORT_PERIOD_RECORDS)
    prev_r = np.maximum.accumulate(np.where(long_run, idx, -1))
    next_r = np.minimum.accumulate(np.where(long_run, idx, nr)[::-1])[::-1]
    for near in (prev_r, next_r):
        has = (near >= 0) & (near < nr)
        r = np.where(has, run_group[np.clip(near, 0, nr - 1)], 1)
        holes = has & (run_group > r) & (run_group % r == 0)
        m = np.where(holes, run_group // r, 2)
        need = np.where(holes, np.maximum(need, _min_period_records(m)), need)
    return ~long_run & (run_group > 0) & (lengths >= need)


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
    """
    if verbose:
        info("Sort ascending: OK", verbose=verbose)
    data = data.sort_index()
    return data


def remove_rows_nat(df: Union[Series, DataFrame], verbose: bool = False) -> tuple[Union[Series, DataFrame], int]:
    """
    Remove rows that do not have a timestamp (NaT).

    Identifies and removes any rows with missing timestamp values (NaT) in the index.
    With *verbose*, warns if more than 10% of rows are removed.

    Parameters
    ----------
    df : Union[Series, DataFrame]
        Data with timestamp index.
    verbose : bool, optional
        Print status messages. Default is False.

    Returns
    -------
    tuple[Union[Series, DataFrame], int]
        - Union[Series, DataFrame]: Data with NaT rows removed. Empty if all
          timestamps are NaT.
        - int: Number of rows removed
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
    """
    current_time_unix = time.time_ns()
    return current_time_unix


def current_datetime(str_format: str = '%Y-%m-%d %H:%M:%S') -> tuple[dt.datetime, str]:
    """
    Current datetime as datetime and string
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime(str_format)
    return now_time_dt, now_time_str


def current_date_str_condensed() -> str:
    """
    Current date as string
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_datetime_str_condensed() -> str:
    """
    Current datetime as string
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d%H%M%S")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_time_microseconds_str() -> str:
    """
    Current time including microseconds as string
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def make_run_id(prefix: str = False) -> str:
    """
    Create string identifier that includes current datetime
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
        verbose (int): Console verbosity. At 2 (or ``True``) and above, prints
            the list of added columns.

    Returns:
        DataFrame: A copy of the dataframe with the new columns appended. With
        *year* enabled, the combined columns ``.YEARMONTH`` (YYYYMM), ``.YEARDOY``
        (YYYYDDD) and ``.YEARWEEK`` (YYYYWW) are added for each of *month*,
        *doy* and *week* that is also enabled.
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
        # Cast to float: insert_season returns nullable Int64 (to carry NaN for months
        # not assigned to a season), and one such column turns the whole frame's
        # .to_numpy() into object dtype, which every ML fit downstream then has to
        # convert back. Float keeps NaN representable, unlike a plain int cast.
        df[season_col] = insert_season(timestamp=df.index).astype(float)
        df[season_sin_col] = np.sin(2 * np.pi * df[season_col] / season_period)
        df[season_cos_col] = np.cos(2 * np.pi * df[season_col] / season_period)
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
        info(f"++ Added new columns with timestamp info: {newcols} {txt}", verbose=verbose)

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
    with N=1000 and progressively reducing to N=4. If both match, the frequency is
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
    - Requires at least 8 rows (4 from start, 4 from end) for detection
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
          appears in >50% of all intervals.
        - freqinfo : str
          Detection result with statistics (e.g., '100% occurrence'),
          '-not-enough-datarows-' if there is no interval to measure (fewer than
          2 timestamps, or none of the deltas is valid), or '-failed-' if no
          interval dominates.

    Notes
    -----
    - Requires at least 2 timestamps (to calculate one interval)
    - Robust to occasional irregular intervals
    - Returns frequency only if most common interval covers >50% of all intervals
      (n timestamps give n-1 intervals)
    - Useful for data with small gaps or timing variations

    References
    ----------
    - https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
    - https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
    """
    inferred_freq, freqinfo, _ = _infer_freq_from_timedelta(timestamp_ix=timestamp_ix)
    return inferred_freq, freqinfo


def _infer_freq_from_timedelta(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """Same as *timestamp_infer_freq_from_timedelta*, plus the exact match fraction.

    Returns (inferred_freq, freqinfo, match_fraction), where *match_fraction* is
    the share of intervals equal to the detected one (0-1), or None if no
    frequency was detected. `freqinfo` formats that fraction with zero decimals,
    so it cannot be the source of the number: a genuine 99.9% would be reported
    as 100%, indistinguishable from a perfectly regular record.
    """
    inferred_freq = None
    freqinfo = None
    df = pd.DataFrame(columns=['tvalue'])
    df['tvalue'] = timestamp_ix
    df['tvalue_shifted'] = df['tvalue'].shift()
    df['delta'] = (df['tvalue'] - df['tvalue_shifted'])
    # n timestamps yield n-1 intervals, because the first delta is always NaT.
    # Dividing by the row count instead under-reports the match (99% for a
    # perfectly regular 100-row index) and rejects a clean 2-row series,
    # because 1 of 2 is not > 50%.
    n_deltas = len(df) - 1  # Number of intervals between timestamps
    detected_deltas = df['delta'].value_counts()  # Found unique deltas
    if n_deltas < 1 or detected_deltas.empty:
        # A single timestamp has no interval at all, and an all-NaT delta column
        # has no mode, so .mode()[0] would raise a bare KeyError here.
        freqinfo = '-not-enough-datarows-'
        return inferred_freq, freqinfo, None
    most_frequent_delta = df['delta'].mode()[0]  # Delta with most occurrences
    most_frequent_delta_counts = detected_deltas[
        most_frequent_delta]  # Number of occurrences for most frequent delta
    most_frequent_delta_perc = most_frequent_delta_counts / n_deltas  # Fraction
    # Check whether the most frequent delta appears in >50% of all intervals
    if most_frequent_delta_perc > 0.50:
        inferred_freq = to_offset(most_frequent_delta)
        inferred_freq = inferred_freq.freqstr
        # inferred_freq = timedelta_to_string(most_frequent_delta)
        freqinfo = f'{most_frequent_delta_perc * 100:.0f}% occurrence'
        # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
        return inferred_freq, freqinfo, most_frequent_delta_perc
    # if most_frequent_delta_perc > 0.90:
    #     inferred_freq = to_offset(most_frequent_delta)
    #     inferred_freq = inferred_freq.freqstr
    #     # inferred_freq = timedelta_to_string(most_frequent_delta)
    #     freqinfo = '>90% occurrence'
    #     # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
    #     return inferred_freq, freqinfo
    else:
        freqinfo = '-failed-'
        return inferred_freq, freqinfo, None


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


def _grid_start(index: DatetimeIndex, freq) -> pd.Timestamp:
    """Return the first timestamp of the regular grid for *index* at *freq*.

    The grid follows the position within the time step that most records
    share, so a single stray or late record cannot move it. Ties go to the
    position of the first record. Calendar frequencies (for example month
    start) have no fixed step, so their grid starts at the first record.

    The position is measured in absolute time because ``pd.date_range``
    builds the grid in absolute time. For time-zone-aware data with a step of
    whole days, the position is measured in wall-clock time.
    """
    try:
        step = freq_as_timedelta(freq, unit=index.unit)
    except ValueError:
        return index[0]
    if index.tz is not None and step % pd.Timedelta(days=1) == pd.Timedelta(0):
        # Daily records at local midnight share one wall-clock position across
        # daylight saving changes. The grid is built in absolute time, so the
        # records after a daylight saving change are not on it.
        wall = index.tz_localize(None)
        position = (wall - wall[0]) % step
    else:
        # After a daylight saving change, the wall-clock times of 2-h, 3-h,
        # 6-h or 12-h records move by one hour. If that position won the vote,
        # the absolute grid would miss every record.
        position = (index - index[0]) % step
    counts = position.value_counts()
    # Keep the unit of the index (a nanosecond step would change it)
    return (index[0] + counts[counts == counts.max()].index.min()).as_unit(index.unit)


def _regularize(data: Union[Series, DataFrame], freq, verbose=False) -> tuple:
    """Reindex *data* to a regular grid at *freq*. Return the data and the
    number of records that are not on the grid and were removed."""
    idx_name = data.index.name

    # Timestamp index between the first and last grid point
    _index = pd.date_range(start=_grid_start(data.index, freq), end=data.index[-1], freq=freq)

    off_grid = ~data.index.isin(_index)
    n_off_grid = int(off_grid.sum())
    if n_off_grid:
        # verbose=False only turns off progress messages. The loss of records
        # is always reported, unless the global verbosity is silent.
        warn(f"Regularize: {n_off_grid} records are not on the {freq} time grid and were removed "
             f"(first: {data.index[off_grid][0]})", verbose=verbose if verbose else None)

    data = data.reindex(_index)
    data.index.name = idx_name

    # Set freq
    data.index = pd.to_datetime(data.index)
    data = data.asfreq(freq=freq)
    if verbose:
        info("Regularize gaps: OK", verbose=verbose)
    return data, n_off_grid


def continuous_timestamp_freq(data: Union[Series, DataFrame], freq: str, verbose: bool = False) -> Union[
    Series, DataFrame]:
    """
    Generate continuous timestamp index without gaps.

    Creates a regular timestamp grid from first to last record at the specified
    frequency, then reindexes data to this grid. Gaps in the original data are
    filled with NaN rows. Useful for time series analysis that requires uniform
    timesteps.

    The grid follows the records: it uses the position within the time step that
    most records share. For example, 30-min data at 00:30, 01:00, ... keep their
    timestamps even if the first record is at 00:00:01. Records that are not on
    the grid (a stray or late record, or 1-min records in 10-min data) are
    removed. Their number is shown as a warning.

    Parameters
    ----------
    data : Union[Series, DataFrame]
        Data with timestamp index.
    freq : str
        Target frequency string (e.g., '30min', '1h', '1D'). Must be a valid pandas
        frequency string.
    verbose : bool, optional
        Print status message. Default is False. The warning about removed
        records is shown also with False, unless the global verbosity is
        silent (see ``diive.set_verbosity``).

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
    - Calendar frequencies (for example 'MS') have no fixed step. Their grid
      starts at the first record.
    - The grid is built in absolute time. Every timestamp of a dataset must use
      the same UTC offset: data whose offset changes within the dataset are
      not supported.
    """
    return _regularize(data=data, freq=freq, verbose=verbose)[0]


def _detect_freq_or_none(index: DatetimeIndex) -> Optional[str]:
    """Time resolution of *index*, None with fewer than two records or none found."""
    if getattr(index, "freq", None) is not None:
        return index.freqstr
    if len(index) < 2:
        return None
    try:
        return DetectFrequency(index).get()
    except (RuntimeError, ValueError):
        return None


def _step_ns(freq: str) -> Optional[int]:
    """Length of a fixed time step in ns, None for calendar steps (month, year)."""
    try:
        return int(to_offset(freq).nanos)
    except ValueError:
        return None


def _freq_text(freq: str) -> str:
    """Readable resolution, e.g. '30min', '10min', '1h'."""
    offset = to_offset(freq)
    return f"{offset.n}{offset.name}"


def _off_grid(index: DatetimeIndex, step: int, origin) -> np.ndarray:
    """Mask of the records of *index* that are not on the grid of *step* ns
    through the timestamp *origin*."""
    ns = index.as_unit("ns").asi8
    return (ns - pd.Timestamp(origin).as_unit("ns").value) % step != 0


def common_time_grid(current: DatetimeIndex, new: DatetimeIndex) -> Optional[str]:
    """Check that records of *new* can join the records of *current*.

    All records of a dataset share one time resolution and one time grid. Use
    this before merging two datasets, then put the merged data on one grid with
    :func:`continuous_timestamp_freq`.

    Args:
        current: Timestamp index of the existing dataset.
        new: Timestamp index of the data to add.

    Returns:
        The shared time resolution as a pandas frequency string, or None when it
        cannot be detected (fewer than two records, or a calendar step such as
        months). With None, nothing could be checked.

    Raises:
        ValueError: The two have different time resolutions, *current* already
            has records off its own time grid, or records of *new* are off the
            time grid of *current*. The message says which and what to do.
    """
    freq_old, freq_new = _detect_freq_or_none(current), _detect_freq_or_none(new)
    if freq_old and freq_new and _step_ns(freq_old) != _step_ns(freq_new):
        raise ValueError(
            f"The current data have a time resolution of {_freq_text(freq_old)}, "
            f"the data to add have {_freq_text(freq_new)}.\n\n"
            "All records of a dataset must have the same time resolution. "
            f"Resample the data to {_freq_text(freq_old)} first, then add them.")
    freq = freq_old or freq_new
    step = _step_ns(freq) if freq else None
    if step is None or len(current) == 0:
        return freq
    origin = current[0]
    if _off_grid(current, step, origin).any():
        raise ValueError(
            "The current data mix time resolutions or have records off their "
            "time grid, so no data can be added to them.\n\n"
            "Load all files again with the same time resolution.")
    off = _off_grid(new, step, origin)
    if off.any():
        raise ValueError(
            f"{int(off.sum()):,} timestamps of the data to add are not on the "
            f"{_freq_text(freq)} time grid of the current data (first: "
            f"{new[off][0]}).\n\n"
            "The added data must have the same time resolution and the same time "
            "steps, e.g. both at :00 and :30 for 30-min data.")
    return freq


def freq_as_timedelta(freq, unit: str, divisor: int = 1) -> pd.Timedelta:
    """Duration of the fixed frequency *freq*, divided by *divisor*, in *unit*.

    A Day offset has no to_timedelta conversion in pandas 3, so the duration is
    taken from its nanoseconds. Pass the unit of the index the result is added
    to: a nanosecond Timedelta would turn a microsecond index into nanoseconds.
    Where the duration does not fit *unit* exactly it stays in nanoseconds.
    """
    timedelta = pd.Timedelta(to_offset(freq).nanos, unit='ns') / divisor
    try:
        return timedelta.as_unit(unit, round_ok=False)
    except ValueError:
        return timedelta


def insert_timestamp(
        data: DataFrame or Series,
        convention: Literal['start', 'middle', 'end'],
        insert_as_first_col: bool = True,
        verbose: bool = False,
        set_as_index: bool = False) -> DataFrame:
    """
    Insert timestamp column that shows the START, END or MIDDLE time of the averaging interval

    The new timestamp column is added as data column, the current
    *data* index remains unchanged. A DataFrame is not copied: the column is
    also added to the DataFrame that was passed in.

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
    timedelta = freq_as_timedelta(timestamp_freq, unit=data.index.unit)
    timedelta_half = freq_as_timedelta(timestamp_freq, unit=data.index.unit, divisor=2)

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


# Reserved timestamp names and the averaging-interval convention each one implies.
TIMESTAMP_CONVENTION_NAMES = {
    'TIMESTAMP_START': 'start',
    'TIMESTAMP_MIDDLE': 'middle',
    'TIMESTAMP_END': 'end',
}


def validate_timestamp_column_name(name: str, convention: Literal['start', 'middle', 'end']) -> None:
    """
    Reject a reserved timestamp name that contradicts the column's convention.

    The names 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE' and 'TIMESTAMP_END' are
    reserved: each one states which point of the averaging interval its timestamps
    mark. Naming a column with one of them while its values follow a *different*
    convention would mislabel the data (and could be misread downstream), so it is
    not allowed. Any other name is accepted.

    Args:
        name: The intended column name.
        convention: The convention the column's timestamps actually follow
            ('start', 'middle' or 'end').

    Raises:
        ValueError: If *name* is a reserved timestamp name for a different
            convention than *convention*.
    """
    implied = TIMESTAMP_CONVENTION_NAMES.get(name)
    if implied is not None and implied != convention:
        raise ValueError(
            f"Column name '{name}' is reserved for the '{implied}' convention, but "
            f"the column follows the '{convention}' convention. Either set "
            f"convention='{implied}' or choose a different column name."
        )


def format_timestamp(
        data: Union[DataFrame, Series, DatetimeIndex],
        convention: Literal['start', 'middle', 'end'],
        fmt: Optional[str] = None,
        verbose: bool = False) -> Series:
    """
    Build a timestamp Series for a chosen averaging-interval convention.

    Derives a START, MIDDLE or END timestamp from *data*'s (timestamp) index
    without modifying that index. The returned Series is aligned to the input
    index, so it can be added to *data* as a new data column while the index
    keeps its own convention (typically TIMESTAMP_MIDDLE).

    The START/END shift reuses :func:`insert_timestamp`, so the input index must
    be a properly named, regular timestamp index ('TIMESTAMP_START',
    'TIMESTAMP_MIDDLE' or 'TIMESTAMP_END') with a known frequency.

    Args:
        data: Source whose index supplies the timestamps. A DataFrame/Series
            (its index is used) or a DatetimeIndex directly.
        convention: Convention of the produced timestamps:
            - 'start': start of the averaging interval
            - 'middle': middle of the averaging interval
            - 'end': end of the averaging interval
        fmt: Optional ``strftime`` format string (e.g. ``'%Y%m%d%H%M'`` for the
            FLUXNET convention). If given, the Series holds formatted strings; if
            ``None``, it holds ``datetime64`` values.
        verbose: If *True*, gives additional text output.

    Returns:
        A Series named 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE' or 'TIMESTAMP_END',
        aligned to *data*'s index.

    Examples:
        >>> import diive as dv, pandas as pd
        >>> idx = pd.date_range('2024-06-01 00:15', periods=4, freq='30min', name='TIMESTAMP_MIDDLE')
        >>> df = pd.DataFrame({'TA': range(4)}, index=idx)
        >>> df['TS'] = dv.times.format_timestamp(df, convention='start', fmt='%Y%m%d%H%M')

    Added in: v0.92.0
    """
    index = data.index if isinstance(data, (DataFrame, Series)) else data
    # Build a fresh index-only frame so insert_timestamp does not mutate *data*.
    frame = insert_timestamp(pd.DataFrame(index=index), convention=convention,
                             insert_as_first_col=False, verbose=verbose)
    col = f"TIMESTAMP_{convention.upper()}"
    series = frame[col]
    if fmt:
        series = series.dt.strftime(fmt)
        series.name = col
    return series


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
        timedelta = freq_as_timedelta(timestamp_freq, unit=data.index.unit, divisor=2)
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
                returned dataset still contains all years, but the data of each
                year end on 11 August (11 August 2016, 11 August 2017 and
                11 August 2018).

    Returns:
        Data with all data after *yearly_end_date* removed
    """
    month = int(yearly_end_date[0:2])
    dayinmonth = int(yearly_end_date[3:])
    data = data.copy()
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

    See Also:
        examples/times/times_keep_daterange.py — date-range subselection examples
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
             f"in date range [{start_ts}, {end_ts}].", verbose=verbose)
    return out


def calc_doy_timefraction(input_series: Series) -> DataFrame:
    """Return a DataFrame with year, day-of-year and intra-day time fraction per timestamp."""
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
    """Return cumulative sums by day-of-year time, one column per year."""
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
