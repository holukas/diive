"""
LONG GAPS: ONEFLUX REMOVAL OF LONG GAP-FILLED PERIODS
=====================================================

Find the records in long gaps that ONEFlux sets missing in its FLUXNET product.

Part of the diive library: https://github.com/holukas/diive

Since 2025, the ONEFlux step that writes the FLUXNET product
(``oneflux/pipeline/site_data_product.py``) sets gap-filled values in long
gaps back to missing. It looks at the QC of each gap-filled variable (0
measured, 1 to 3 the quality class of the MDS fill) and finds the runs of
records with QC above 2, the poorest class. A run shorter than 15 days stays
filled. In the middle of the record a run must be longer than 2 x 15 + 5
days, and then only its inner part goes: 15 days stay filled at each end. A
run at the start or the end of the record is removed except for the last 15
days minus one record next to the measured data. The variable and its QC become missing there, and so do the
variables that ONEFlux derives from it (GPP and RECO of the same NEE
version). Daily, weekly, monthly and yearly values are missing when their
period holds one such record.

:func:`oneflux_long_gap_mask` reproduces ``get_indices_to_filter`` and
:func:`oneflux_long_gap_periods` the aggregation step of
``generate_agg_timestamp_mask``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: ONEFlux's settings in ``site_data_product.run_site``: QC above 2 counts,
#: runs are checked against 15 days, a run in the middle of the record must
#: be longer than 2 x 15 + 5 days.
ONEFLUX_LONG_GAP_QC_THRESHOLD = 2
ONEFLUX_LONG_GAP_WINDOW_DAYS = 15
ONEFLUX_LONG_GAP_MINIMUM_GAP_DAYS = 5


def _records_per_day(index: pd.DatetimeIndex) -> int:
    """Records per day from the median time step of a DatetimeIndex."""
    if len(index) < 2:
        raise ValueError("Need at least two records to know the time step, or give "
                         "records_per_day.")
    step = pd.Series(index).diff().median()
    return int(round(pd.Timedelta('1D') / step))


def oneflux_long_gap_mask(qc, *, qc_threshold: float = ONEFLUX_LONG_GAP_QC_THRESHOLD,
                          window_days: float = ONEFLUX_LONG_GAP_WINDOW_DAYS,
                          minimum_gap_days: float = ONEFLUX_LONG_GAP_MINIMUM_GAP_DAYS,
                          records_per_day: int | None = None):
    """Records in long gaps that ONEFlux sets missing in its FLUXNET product.

    Port of ``get_indices_to_filter`` in ONEFlux
    ``oneflux/pipeline/site_data_product.py`` (new in 2025). ONEFlux runs it on
    the QC of each gap-filled variable of the half-hourly or hourly product
    (``NEE_CUT_REF_QC``, ``LE_F_MDS_QC``, ...) with the defaults here: 15 days
    are 720 half-hourly or 360 hourly records.

    A record is bad when its QC is above ``qc_threshold``. A missing QC (NaN)
    is not bad. Runs of bad records shorter than the window are kept. For a
    longer run, with ``w`` the window in records and ``first`` and ``last``
    the positions of its first and last record:

    - At the start of the record (``first`` is 0): runs of up to ``w + 1``
      records are kept. A longer run is removed from the start up to and
      including position ``last - w + 1``, so ``w - 1`` records stay next to
      the data.
    - At the end of the record (``last`` is the last position): runs of up
      to ``w + 1`` records are kept. A longer run is removed from position
      ``first + w - 1`` to the end.
    - In the middle: a run of up to ``2 w + g`` records (``g`` the minimum gap
      in records) is kept. In a longer run the records from ``first + w`` to
      ``last - w`` are removed, ``w`` records stay at each end.

    These rules, including the asymmetric ends, are ONEFlux's.

    Args:
        qc: QC of a gap-filled variable: 0 measured, 1 to 3 the MDS quality
            class, NaN unknown. A Series with a DatetimeIndex, or an array.
            The record start and end are the first and last value: ONEFlux
            runs on full calendar years, so give full years to get its result.
        qc_threshold: QC above this value is bad (ONEFlux: 2).
        window_days: Window in days (ONEFlux: 15).
        minimum_gap_days: Extra length in days a run in the middle of the
            record needs on top of the two windows (ONEFlux: 5).
        records_per_day: Records per day (48 half-hourly, 24 hourly). None
            takes it from the median time step of the Series index.

    Returns:
        Boolean mask, True for the records to set missing: a Series on the
        index of ``qc`` if a Series was given, else an array.
    """
    is_series = isinstance(qc, pd.Series)
    if records_per_day is None:
        if not (is_series and isinstance(qc.index, pd.DatetimeIndex)):
            raise ValueError("records_per_day is needed when qc has no DatetimeIndex.")
        records_per_day = _records_per_day(qc.index)
    window = int(round(window_days * records_per_day))
    minimum_gap = int(round(minimum_gap_days * records_per_day))
    if window < 1:
        raise ValueError(f"The window must be at least one record, got {window}.")

    values = np.asarray(qc, dtype=float)
    n = values.size
    with np.errstate(invalid='ignore'):
        bad = values > qc_threshold

    # ONEFlux finds the windows of w bad records (a convolution) and merges
    # the overlapping ones. The merged blocks are exactly the runs of bad
    # records of length w or more, which are found here directly.
    edges = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    mask = np.zeros(n, dtype=bool)
    for first, last in zip(starts, ends, strict=True):
        if last - first + 1 < window:
            continue
        if first == 0:
            # Start of the record
            if last > window:
                mask[first:last - window + 2] = True
        elif last == n - 1:
            # End of the record
            if first < n - 1 - window:
                mask[first + window - 1:last + 1] = True
        elif (last - first + 1) > (2 * window + minimum_gap):
            # Middle of the record
            mask[first + window:last - window + 1] = True

    if is_series:
        return pd.Series(mask, index=qc.index, name=qc.name)
    return mask


def oneflux_long_gap_periods(record_start: pd.DatetimeIndex, first_day, last_day) -> np.ndarray:
    """Periods that hold a record removed as long gap, as ONEFlux finds them.

    Port of ``generate_agg_timestamp_mask`` in ONEFlux
    ``oneflux/pipeline/site_data_product.py``. ONEFlux takes the dates of the
    removed half-hourly (or hourly) records from their TIMESTAMP_START and
    sets a daily, weekly, monthly or yearly value missing when its period
    holds one of these dates.

    Args:
        record_start: TIMESTAMP_START of the removed records.
        first_day: First day of each period.
        last_day: Last day of each period (the same as ``first_day`` for
            daily values). ONEFlux weeks run 7 days from 1 January, the
            last week of a year takes the remaining 8 or 9 days.

    Returns:
        Boolean array, True for the periods to set missing.
    """
    days = np.unique(pd.DatetimeIndex(record_start).normalize().to_numpy())
    first = pd.DatetimeIndex(first_day).normalize().to_numpy()
    last = pd.DatetimeIndex(last_day).normalize().to_numpy()
    if len(first) != len(last):
        raise ValueError("first_day and last_day need the same length.")
    lo = np.searchsorted(days, first, side='left')
    hi = np.searchsorted(days, last, side='right')
    return hi > lo
