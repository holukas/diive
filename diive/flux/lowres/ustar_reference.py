"""
USTAR REFERENCE: ONEFLUX REF THRESHOLD SELECTED BY MODEL EFFICIENCY
===================================================================

Select the reference u* threshold (REF) among gap-filled NEE versions, as
ONEFlux ``nee_proc`` 1.3.7 does for ``NEE_CUT_REF`` and ``NEE_VUT_REF``.

Part of the diive library: https://github.com/holukas/diive

ONEFlux filters NEE with the u* thresholds at 40 percentiles of the bootstrap
(1.25 to 98.75 in steps of 2.5) and gap-fills each version with MDS. The REF is
the version most similar to all the others: for each version ``j`` the score is
the sum over all versions ``i`` of the model efficiency
``1 - MSE(j, i) / var(j)``, and the version with the highest score is the REF.
ONEFlux selects it five times, on half-hourly (HH), daily (DD), weekly (WW),
monthly (MM) and yearly (YY) values, so each time resolution of a FLUXNET
product can have its own REF. The 50th percentile version (USTAR50) is not a
candidate.

The code reproduces ``nee_proc`` (``oneflux_steps/nee_proc/src/dataset.c``,
cited as ``dataset.c``) including its quirks:

- Before scoring, long gaps filled with low quality are left out
  (``create_nee_matrix_for_ref``, dataset.c:714-979). A record is "bad" when the
  gap-filling quality of the version with the highest percentile is above
  ``mef_qc`` (2, so quality 3 only). Inside a run of bad records, the records
  more than ``mef_max_gap`` records (720) away from a good record on both sides
  are removed in all versions, if they form a block of at least
  ``mef_min_gap`` records (240). The run counters start at ``mef_max_gap + 1``
  at the first and the last record whatever their quality, so a run starting
  at the second record counts as touching the start (dataset.c:779, 795). The
  last record is never removed (dataset.c:847). These are record counts, so
  for hourly data they mean 30 and 10 days. Measured records have quality 0,
  records MDS could not fill have no quality and count as good.
- The daily values are the mean of the records of the day times ``CO2TOC``
  (1.03772448, umol m-2 s-1 to gC m-2 d-1), missing when any record of the day
  is missing (dataset.c:5996-6045). Weekly values are the mean of 7 days, the
  52nd week of a year takes the last 8 or 9 days (dataset.c:7077-7180).
  Monthly values are the mean of the days of the month (dataset.c:8770-8830),
  yearly values the sum of the days (dataset.c:9930-9975). Any missing day
  makes the week, month or year missing.
- Every sum runs in record order, as in the C (dataset.c:1036-1078), so the
  scores are bitwise equal to the C on the same input.
- A value counts as missing when ``(int)value == -9999`` (``IS_INVALID_VALUE``,
  common.h:40), so values in (-10000, -9999] are missing too.
- Ties go to the lowest percentile (strict ``>``, dataset.c:1097).
- With a single year, the yearly REF is the monthly REF (dataset.c:1266).
  diive applies this to CUT and VUT. ONEFlux reports CUT only for 3 or more
  years (dataset.c:4204), so it never meets a single-year CUT.
- For VUT, ONEFlux picks one REF percentile for all years. Each year then
  uses its own threshold at that percentile.

What the selection needs as input is the NEE matrix ``nee_proc`` builds: per
version the measured NEE where it survived the u* filter, else the MDS
prediction, and its MDS quality (0 measured, 1-3 filled). In diive these are
the L4.1 MDS outputs of the u* scenarios. ``FluxMDS(mode='oneflux')``
reproduces the gap-filling of ``nee_proc``. Note that ONEFlux also removes the
first record after a period below the threshold, diive does not (finding L74).

GPP and RECO get their own REF in the ONEFlux step ``ure``, from the
partitioning of each of the 40 versions:
:func:`select_partitioning_reference` reproduces it, with the same model
efficiency but its own rules for missing values, and
:func:`oneflux_ure_cleanup` the cleaning of the partitioning results before
it. ``ure`` also gives the USTAR50 version and the mean, standard error and 7
percentiles across the versions (``GPP_NT_VUT_MEAN``, ``_SE``, ``_05`` to
``_95``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from diive.flux.lowres._ustar_mp_kernels import _jit, is_invalid

#: The 40 u* percentiles of ONEFlux ``nee_proc`` that compete for REF
#: (``percentiles_test_2``, dataset.c:77-118, without its last entry 50).
ONEFLUX_REF_PERCENTILES: tuple[float, ...] = tuple(1.25 + 2.5 * i for i in range(40))

#: Time resolutions of the REF selection, in the order ONEFlux runs them.
REF_AGGREGATIONS: tuple[str, ...] = ('HH', 'DD', 'WW', 'MM', 'YY')

#: nee_proc defaults (main.c:57-59).
ONEFLUX_MEF_QC = 2
ONEFLUX_MEF_MIN_GAP = 5 * 48
ONEFLUX_MEF_MAX_GAP = 15 * 48

#: umol CO2 m-2 s-1 to g C m-2 d-1 (common.h:85).
CO2TOC = 1.03772448

_INVALID = -9999.0


# ------------------------------------------------------------------ kernels

@_jit
def _mef_mask_kernel(bad, max_gap, min_gap, mask, blocks):
    """Port of the filter in ``create_nee_matrix_for_ref`` (dataset.c:771-929).

    ``bad`` is True where the quality of the highest-percentile version is
    above ``mef_qc``. Sets ``mask`` (records left out) and writes one row
    ``(start, end, applied)`` per candidate block to ``blocks`` (end exclusive).
    Returns the number of blocks.
    """
    n = bad.shape[0]
    fwd = np.empty(n, np.int64)
    bkw = np.empty(n, np.int64)
    s = max_gap + 1
    fwd[0] = s
    for i in range(1, n):
        if bad[i]:
            s += 1
        else:
            s = 0
        fwd[i] = s
    s = max_gap + 1
    bkw[n - 1] = s
    for i in range(n - 2, -1, -1):
        if bad[i]:
            s += 1
        else:
            s = 0
        bkw[i] = s

    nb = 0
    start = -1
    for i in range(n):
        if fwd[i] > max_gap and bkw[i] > max_gap and i != n - 1:
            if start == -1:
                start = i
        elif start >= 0:
            end = i
            applied = (end - start) >= min_gap
            if applied:
                for y in range(start, end):
                    mask[y] = True
            blocks[nb, 0] = start
            blocks[nb, 1] = end
            blocks[nb, 2] = 1 if applied else 0
            nb += 1
            start = -1
    return nb


@_jit
def _get_reference_kernel(m, mes, mess):
    """Port of ``get_reference`` (dataset.c:982-1140).

    ``m`` has one row per version (k x n records), -9999 = missing. Fills
    ``mes[i, j] = 1 - MSE(j, i) / var(j)`` (-9999 where versions i and j share
    no record) and ``mess[j]`` (score of version j). Returns the REF index, or
    -1 where ONEFlux fails.
    """
    k, n = m.shape
    for i in range(k):
        for j in range(k):
            mes[i, j] = _INVALID
    for column in range(k):
        mean = 0.0
        mean_count = 0
        for row in range(n):
            v = m[column, row]
            if not is_invalid(v):
                mean += v
                mean_count += 1
        if mean_count:
            mean /= mean_count
        square = 0.0
        square_count = 0
        for row in range(n):
            v = m[column, row]
            if not is_invalid(v):
                square += (v - mean) * (v - mean)
                square_count += 1
        if square_count:
            variance = square / square_count
        else:
            return -2  # "unable to compute variance for column", dataset.c:1054
        for i in range(k):
            s = 0.0
            valid = 0
            for j in range(n):
                a = m[column, j]
                b = m[i, j]
                if not is_invalid(a) and not is_invalid(b):
                    s += (a - b) * (a - b)
                    valid += 1
            if valid:
                s /= valid
                s /= variance
                mes[i, column] = 1 - s
            else:
                mes[i, column] = _INVALID
    for column in range(k):
        mess[column] = 0.0
        for row in range(k):
            if not is_invalid(mes[row, column]):
                mess[column] += mes[row, column]
    column = 0
    s = mess[0]
    for i in range(1, k):
        if mess[i] > s:
            s = mess[i]
            column = i
    # nee_proc then fails on an invalid best score (dataset.c:1103), ure does
    # not (ure dataset.c:714-775): the caller checks.
    return column


@_jit
def _daily_kernel(hh, rows_per_day, co2toc):
    """Daily values for the REF selection (dataset.c:5996-6045)."""
    k, n = hh.shape
    nd = n // rows_per_day
    out = np.empty((k, nd))
    for p in range(k):
        for d in range(nd):
            s = 0.0
            has_invalid = False
            for i in range(rows_per_day):
                v = hh[p, d * rows_per_day + i]
                if is_invalid(v):
                    has_invalid = True
                s += v
            if has_invalid:
                out[p, d] = _INVALID
            else:
                s /= rows_per_day
                s *= co2toc
                out[p, d] = s
    return out


@_jit
def _window_kernel(dd, starts, lengths, divide):
    """Mean (``divide``) or sum of the daily values in each window ``starts[g]``
    to ``starts[g] + lengths[g]``, missing if any day is missing."""
    k = dd.shape[0]
    ng = lengths.shape[0]
    out = np.empty((k, ng))
    for p in range(k):
        for g in range(ng):
            pos = starts[g]
            z = lengths[g]
            s = 0.0
            has_invalid = False
            for j in range(z):
                v = dd[p, pos + j]
                if is_invalid(v):
                    has_invalid = True
                s += v
            if has_invalid:
                out[p, g] = _INVALID
            else:
                if divide:
                    s /= z
                out[p, g] = s
    return out


@_jit
def _group_kernel(dd, lengths, divide):
    """Weekly or monthly means (``divide``) or yearly sums of the daily values,
    in consecutive groups of ``lengths`` days."""
    starts = np.zeros(lengths.shape[0], np.int64)
    for g in range(1, lengths.shape[0]):
        starts[g] = starts[g - 1] + lengths[g - 1]
    return _window_kernel(dd, starts, lengths, divide)


# ------------------------------------------------------------------ helpers

def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def _week_lengths(years) -> np.ndarray:
    """52 weeks per year: 51 of 7 days, the last one takes the rest (8 or 9)."""
    out = []
    for y in years:
        ndays = 366 if _is_leap(y) else 365
        out += [7] * 51 + [ndays - 357]
    return np.array(out, dtype=np.int64)


def _month_lengths(years) -> np.ndarray:
    dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    out = []
    for y in years:
        out += [d + (1 if (i == 1 and _is_leap(y)) else 0) for i, d in enumerate(dpm)]
    return np.array(out, dtype=np.int64)


def _year_lengths(years) -> np.ndarray:
    return np.array([366 if _is_leap(y) else 365 for y in years], dtype=np.int64)


def _on_year_grid(frame: pd.DataFrame):
    """``frame`` on the full-year TIMESTAMP_MIDDLE grid of every calendar year
    from its first to its last, as the yearly ONEFlux files hold the records.

    Returns ``(grid, years, rows_per_day)``.
    """
    from diive.flux.lowres.ustar_detection import _full_year, _record_step, _to_middle
    frame = _to_middle(frame).sort_index()
    step = _record_step(frame.index)
    rows_per_day = int(round(pd.Timedelta('1D') / step))
    if rows_per_day not in (24, 48):
        raise ValueError(f"Records must be half-hourly or hourly, got a step of {step}.")
    years = list(range(int(frame.index.year.min()), int(frame.index.year.max()) + 1))
    grid = pd.concat([_full_year(frame, y, step) for y in years])
    return grid, years, rows_per_day


def oneflux_ref_mask(qc, *, mef_qc: int = ONEFLUX_MEF_QC,
                     mef_min_gap: int = ONEFLUX_MEF_MIN_GAP,
                     mef_max_gap: int = ONEFLUX_MEF_MAX_GAP):
    """Records ONEFlux leaves out of the REF selection.

    Args:
        qc: MDS quality of the version with the highest percentile, one value
            per record of the full-year grid (0 measured, 1-3 filled, NaN not
            filled).
        mef_qc: Highest quality kept (0-3). 3 keeps every record.
        mef_min_gap: Smallest block (records) that is removed.
        mef_max_gap: Distance (records) from a good record beyond which bad
            records are removed.

    Returns:
        ``(mask, blocks)``: boolean array (True = left out) and a list of
        ``(start, end, applied)`` tuples, ``end`` exclusive, for every block
        that met the distance rule, also those shorter than ``mef_min_gap``.
    """
    qc = np.asarray(qc, dtype=float)
    n = len(qc)
    mask = np.zeros(n, dtype=np.bool_)
    if n == 0 or mef_qc >= 3:  # dataset.c:739, no filter with mef_qc 3
        return mask, []
    # (int)-9999 > 2 is False in the C, NaN > 2 is False here.
    bad = qc > mef_qc
    blocks = np.zeros((n, 3), dtype=np.int64)
    nb = _mef_mask_kernel(bad, int(mef_max_gap), int(mef_min_gap), mask, blocks)
    return mask, [(int(a), int(b), bool(c)) for a, b, c in blocks[:nb]]


def oneflux_model_efficiency(values, *, fail_on_invalid_score: bool = True):
    """Model efficiency scores and REF index as ``get_reference`` (dataset.c:982-1140).

    Args:
        values: 2-D array, one row per record (or aggregated period) and one
            column per version, NaN or -9999 = missing.
        fail_on_invalid_score: Return -1 when the best score is missing
            (``nee_proc``, dataset.c:1103). ONEFlux ``ure`` has no such check
            and returns the best column anyway: pass False for it.

    Returns:
        ``(index, scores, mef)``: the REF column (-1 when ONEFlux fails: a
        version without any value, or no valid score), the score of each
        version (``mess``), and the matrix ``mef[i, j] = 1 - MSE(j, i) /
        var(j)`` (NaN where versions i and j share no value).
    """
    m = np.asarray(values, dtype=float).T.copy()
    m[np.isnan(m)] = _INVALID
    k = m.shape[0]
    mes = np.empty((k, k))
    mess = np.zeros(k)
    index = int(_get_reference_kernel(m, mes, mess))
    if index == -2:
        index = -1
        mess[:] = np.nan
    elif fail_on_invalid_score and -10000.0 < mess[index] <= -9999.0:
        index = -1
    mef = np.where((mes > -10000.0) & (mes <= -9999.0), np.nan, mes)
    return index, mess, mef


# ------------------------------------------------------------------ result

@dataclass(frozen=True)
class UstarReferenceResult:
    """REF selection for one threshold type (CUT or VUT).

    Attributes:
        percentiles: Percentile of each version, in column order.
        index: REF column per time resolution (``'HH'``, ``'DD'``, ``'WW'``,
            ``'MM'``, ``'YY'``), -1 = no REF.
        percentile: REF percentile per time resolution, NaN = no REF.
        scores: Score of each version (columns: percentiles) per time
            resolution (rows). NaN for the yearly row of a single year, which
            takes the monthly REF.
        model_efficiency: Per time resolution, the matrix ``[i, j] = 1 -
            MSE(j, i) / var(j)``. The score of version j is the sum of column j.
        masked: Records left out of the selection (True), on the full-year
            TIMESTAMP_MIDDLE grid.
        masked_blocks: Blocks found by the distance rule: first and last
            record (TIMESTAMP_MIDDLE), number of records, and whether the block
            was long enough to be removed.
        years: Calendar years of the grid.
    """
    percentiles: tuple
    index: dict
    percentile: dict
    scores: pd.DataFrame
    model_efficiency: dict = field(repr=False)
    masked: pd.Series = field(repr=False)
    masked_blocks: pd.DataFrame = field(repr=False)
    years: tuple = ()

    def summary(self) -> str:
        """REF percentile per time resolution as text."""
        lines = ["REF u* percentile (ONEFlux model efficiency)", "=" * 46]
        for agg in REF_AGGREGATIONS:
            p = self.percentile.get(agg, np.nan)
            txt = f"{p:g}" if np.isfinite(p) else "none"
            note = "  (monthly REF, single year)" if (
                agg == 'YY' and len(self.years) == 1) else ""
            lines.append(f"{agg:<4}{txt:>8}{note}")
        n_masked = int(self.masked.sum())
        lines.append(f"Records left out: {n_masked} of {len(self.masked)}")
        return "\n".join(lines)


def select_ustar_reference(nee: pd.DataFrame, qc: pd.DataFrame,
                           percentiles=ONEFLUX_REF_PERCENTILES, *,
                           mef_qc: int = ONEFLUX_MEF_QC,
                           mef_min_gap: int = ONEFLUX_MEF_MIN_GAP,
                           mef_max_gap: int = ONEFLUX_MEF_MAX_GAP) -> UstarReferenceResult:
    """Select the REF u* threshold as ONEFlux ``nee_proc`` does.

    Args:
        nee: Gap-filled NEE (umol m-2 s-1), one column per u* version in the
            order of ``percentiles``: measured where kept after the u* filter,
            else the MDS value, NaN where MDS could not fill. Half-hourly or
            hourly records on a regular DatetimeIndex (TIMESTAMP_MIDDLE, or an
            index named TIMESTAMP_START or TIMESTAMP_END). Each calendar year is
            padded to its full grid with missing values, and missing years
            between the first and the last are added as missing, as ONEFlux
            adds them. ONEFlux would gap-fill such records, so give full years.
        qc: MDS quality of each value, same shape as ``nee``: 0 measured, 1-3
            filled (``FluxMDS`` ``.PREDICTIONS_QUALITY``), NaN not filled.
            Only the column of the highest percentile is used, as ONEFlux uses
            the 98.75 % version (dataset.c:746).
        percentiles: Percentile of each column. Defaults to
            :data:`ONEFLUX_REF_PERCENTILES`.
        mef_qc, mef_min_gap, mef_max_gap: ``nee_proc`` options ``-mef_qc``,
            ``-mef_min_gap`` and ``-mef_max_gap`` (records), see
            :func:`oneflux_ref_mask`.

    Returns:
        :class:`UstarReferenceResult`. When ONEFlux would fail at a time
        resolution before the yearly one, that resolution and all later ones
        have no REF (ONEFlux stops processing the site there).
    """
    percentiles = tuple(float(p) for p in percentiles)
    if nee.shape[1] != len(percentiles):
        raise ValueError(f"nee has {nee.shape[1]} columns but {len(percentiles)} "
                         f"percentiles were given.")
    if qc.shape != nee.shape:
        raise ValueError(f"qc must have the shape of nee {nee.shape}, got {qc.shape}.")
    if len(percentiles) < 2:
        raise ValueError("The REF selection needs at least two versions.")
    if not isinstance(nee.index, pd.DatetimeIndex) or not nee.index.equals(qc.index):
        raise ValueError("nee and qc need the same DatetimeIndex.")

    frame = pd.DataFrame(np.column_stack([nee.to_numpy(float), qc.to_numpy(float)]),
                         index=nee.index)
    frame.index.name = nee.index.name
    grid, years, rows_per_day = _on_year_grid(frame)

    k = len(percentiles)
    hh = grid.iloc[:, :k].to_numpy(float).T.copy()  # versions x records
    hh[np.isnan(hh)] = _INVALID
    qc_top = grid.iloc[:, 2 * k - 1].to_numpy(float)

    mask, blocks = oneflux_ref_mask(qc_top, mef_qc=mef_qc, mef_min_gap=mef_min_gap,
                                    mef_max_gap=mef_max_gap)
    hh_ref = hh.copy()
    hh_ref[:, mask] = _INVALID

    dd = _daily_kernel(hh_ref, rows_per_day, CO2TOC)
    matrices = {
        'HH': hh_ref,
        'DD': dd,
        'WW': _group_kernel(dd, _week_lengths(years), True),
        'MM': _group_kernel(dd, _month_lengths(years), True),
        'YY': _group_kernel(dd, _year_lengths(years), False),
    }

    index, percentile, scores, mefs = {}, {}, {}, {}
    failed = False
    for agg in REF_AGGREGATIONS:
        if failed:
            index[agg], percentile[agg] = -1, np.nan
            scores[agg] = np.full(k, np.nan)
            continue
        if agg == 'YY' and len(years) == 1:
            # dataset.c:1266: the yearly REF stays the monthly one
            index[agg] = index['MM']
            scores[agg] = np.full(k, np.nan)
        else:
            idx, mess, mef = oneflux_model_efficiency(matrices[agg].T)
            index[agg], scores[agg], mefs[agg] = idx, mess, mef
            # A failure stops nee_proc for the site, except at YY (dataset.c:1271)
            if idx == -1 and agg != 'YY':
                failed = True
        percentile[agg] = percentiles[index[agg]] if index[agg] >= 0 else np.nan

    masked = pd.Series(mask, index=grid.index, name='REF_MASKED')
    tsi = grid.index
    blocks_df = pd.DataFrame(
        [{'start': tsi[a], 'end': tsi[b - 1], 'records': b - a, 'applied': applied}
         for a, b, applied in blocks],
        columns=['start', 'end', 'records', 'applied'])
    return UstarReferenceResult(
        percentiles=percentiles,
        index=index,
        percentile=percentile,
        scores=pd.DataFrame([scores[a] for a in REF_AGGREGATIONS],
                            index=list(REF_AGGREGATIONS), columns=list(percentiles)),
        model_efficiency=mefs,
        masked=masked,
        masked_blocks=blocks_df,
        years=tuple(years),
    )


# ================================================================ GPP / RECO
#
# ONEFlux selects the REF of GPP and RECO in its step ``ure`` (Uncertainty and
# References Extraction, ``oneflux_steps/ure/src``, v1.021), cited below as
# ``ure dataset.c`` and ``ure main.c``. Its inputs are written by the pipeline
# step ``prepare_ure`` (``oneflux/pipeline/wrappers.py``).

#: ``ure`` defaults (ure main.c:34-52): a version takes part in the selection
#: when it has at least 90 % of the valid values of the most complete version,
#: at least 10 versions must take part, and the 7 percentiles across the
#: versions need at least 30 valid versions (25, 50 and 75 only: more than 10).
URE_VALID_DATA_COUNT = 90
URE_VALID_PERC_COUNT = 10
URE_MIN_7_PERC_COUNT = 30
URE_MIN_3_PERC_COUNT = 10

#: Percentiles across the 40 versions that ``ure`` reports
#: (``percentiles_test_1``, ure dataset.c:80), as in ``GPP_NT_VUT_05``.
URE_PERCENTILES: tuple[int, ...] = (5, 16, 25, 50, 75, 84, 95)

#: Columns of :attr:`PartitioningReferenceResult.tables`.
URE_TABLE_COLUMNS: tuple[str, ...] = (
    ('REF', 'USTAR50', 'MEAN', 'SE') + tuple(f'{p:02d}' for p in URE_PERCENTILES))

#: Divisor of the standard error, sqrt(40) as a literal (ure dataset.c:1082).
_URE_SQRT40 = 6.324555320336759


@_jit
def _ure_stats_kernel(m, index_table, keep3, min3, min7, out):
    """Statistics across the versions per row, as ``process_matrix`` (ure dataset.c:1035-1086).

    ``m`` is versions x rows (-9999 missing). ``index_table[y, q]`` is the
    position of percentile q in the sorted values when y are valid,
    ``keep3[q]`` is True for 25, 50 and 75. ``out[row]`` gets the 7
    percentiles, the mean and the standard error, and stays NaN where ``ure``
    writes -9999.
    """
    k, n = m.shape
    nq = index_table.shape[1]
    buf = np.empty(k)
    for r in range(n):
        y = 0
        s = 0.0
        for p in range(k):
            v = m[p, r]
            if not is_invalid(v):
                buf[y] = v
                s += v
                y += 1
        if y <= min3:
            continue
        srt = np.sort(buf[:y])
        for q in range(nq):
            if y < min7 and not keep3[q]:
                continue
            out[r, q] = srt[index_table[y, q]]
        # get_mean_allowing_invalid and get_standard_deviation_allowing_invalid
        # (ure dataset.c:858-970): sums in version order.
        mean = s / y
        out[r, nq] = mean
        if mean != mean or is_invalid(mean):
            continue
        s2 = 0.0
        for p in range(y):
            d = buf[p] - mean
            d *= d
            s2 += d
        s2 /= y - 1
        sd = np.sqrt(s2)
        if sd != sd:
            continue
        out[r, nq + 1] = sd / _URE_SQRT40


def _ure_percentile_index(percentile: float, y: int) -> int:
    """Position of ``percentile`` in ``y`` sorted values (``get_percentile_allowing_invalid``,
    ure dataset.c:836-845): ``ROUND(percentile / 100 * y) - 1`` in float, clipped."""
    x = np.float32(np.float32(percentile) / np.float32(100)) * np.float32(y)
    i = int(float(x) + 0.5) - 1  # ROUND (common.h:63) adds 0.5 in double
    return min(max(i, 0), y - 1)


def _ure_week_windows(years, dd_week):
    """First day and length of the 52 weeks of each year, as ``ure`` averages them
    (ure dataset.c:2204-2292).

    Weeks 1-51 have 7 days. Week 52 starts at day 351 plus ``j``, the loop
    counter left over from the last version ``ure`` aggregated in week 51: the
    USTAR50 version (of CUT with 3 or more years, ure dataset.c:2226-2234).
    That loop stops at the first missing day, so ``j`` is the number of valid
    days before it, or 7. With a missing day in that week, week 52 starts
    earlier and overlaps week 51.
    """
    starts, lengths = [], []
    first = 0
    for year in years:
        ndays = 366 if _is_leap(year) else 365
        j = 7
        for d in range(7):
            v = dd_week[first + 350 + d]
            if -10000.0 < v <= -9999.0:
                j = d
                break
        starts += [first + 7 * i for i in range(51)] + [first + 350 + j]
        lengths += [7] * 51 + [ndays - 350 - j]
        first += ndays
    return np.array(starts, dtype=np.int64), np.array(lengths, dtype=np.int64)


def _ure_ref_matrix(m, valid_data_count: int, valid_perc_count: int):
    """The values ``ure`` selects the REF from (``create_matrix_for_ref``, ure
    dataset.c:316-589).

    ``m`` is rows x versions, -9999 missing. Rows where all versions are
    missing are dropped. A version takes part if it has at least
    ``(int)(valid_data_count / 100 * n_max)`` valid values (in float), with
    ``n_max`` the most valid values of any version. With fewer than
    ``valid_perc_count`` versions taking part there is no REF. Then every row
    with a missing value in a taking-part version is dropped.

    Returns ``(values, versions)`` (columns of ``m``), or None for no REF.
    """
    inv = (m > -10000.0) & (m <= -9999.0)
    rows = ~inv.all(axis=1)
    m, inv = m[rows], inv[rows]
    if m.shape[0] == 0:
        return None
    counts = (~inv).sum(axis=0)
    # "used float var due promotion" (ure dataset.c:411-414)
    need = int(np.float32(np.float32(valid_data_count) / np.float32(100))
               * np.float32(int(counts.max())))
    versions = np.flatnonzero(counts >= need)
    if len(versions) == 0 or len(versions) < valid_perc_count:
        return None
    rows = ~inv[:, versions].any(axis=1)
    if not rows.any():
        return None
    return m[rows][:, versions], versions


def oneflux_ure_cleanup(reco, gpp, method: str):
    """Clean RECO and GPP as ONEFlux does before ``ure`` (``prepare_ure``).

    Port of ``check_cleanup_nt`` and ``check_cleanup_dt`` of
    ``PipelinePrepareURE`` in ``oneflux/pipeline/wrappers.py``. A value is
    missing when it is NaN, infinite or at most -6990 (``nan_ext``). Then:

    - ``'nighttime'``: where RECO is missing or outside -100 to 100, RECO and
      GPP are both set missing. A missing RECO is -9999 at that point and so
      below -100, so it removes GPP as well.
    - ``'daytime'``: negative GPP is missing, RECO outside -100 to 100 is
      missing (GPP is kept), GPP outside -150 to 150 is missing.

    ONEFlux also keeps every second record of hourly daytime results written
    on a half-hourly grid. diive's daytime ONEFlux port returns them hourly
    already, so that step is not needed here.

    Args:
        reco: Ecosystem respiration of one u* version (umol m-2 s-1).
        gpp: Gross primary production of the same version.
        method: ``'nighttime'`` or ``'daytime'``.

    Returns:
        ``(reco, gpp)`` with NaN for missing values, as Series if Series
        were given.
    """
    if method not in ('nighttime', 'daytime'):
        raise ValueError(f"method must be 'nighttime' or 'daytime', got {method!r}.")
    nan = -9999.0  # NAN (oneflux/partition/auxiliary.py:25)
    r = np.asarray(reco, dtype=float).copy()
    g = np.asarray(gpp, dtype=float).copy()

    def nan_ext(a):  # auxiliary.py:45-53
        with np.errstate(invalid='ignore'):
            return np.isnan(a) | np.isinf(a) | (a <= -9990.0) | (a <= -6990.0)

    r[nan_ext(r)] = nan
    g[nan_ext(g)] = nan
    if method == 'nighttime':
        bad = (r < -100) | (r > 100)
        r[bad] = nan
        g[bad] = nan
    else:
        g[g < 0] = nan
        r[(r < -100) | (r > 100)] = nan
        g[(g < -150) | (g > 150)] = nan
    r[r == nan] = np.nan
    g[g == nan] = np.nan
    if isinstance(reco, pd.Series):
        r = pd.Series(r, index=reco.index, name=reco.name)
    if isinstance(gpp, pd.Series):
        g = pd.Series(g, index=gpp.index, name=gpp.name)
    return r, g


@dataclass(frozen=True)
class PartitioningReferenceResult:
    """REF selection of one partitioning product (GPP or RECO) and threshold type.

    Attributes:
        percentiles: Percentile of each version, in column order.
        index: REF column per time resolution (``'HH'``, ``'DD'``, ``'WW'``,
            ``'MM'``, ``'YY'``), -1 = no REF.
        percentile: REF percentile per time resolution, NaN = no REF.
        scores: Score of each version (columns: percentiles) per time
            resolution (rows). NaN for versions left out of the selection, and
            for the yearly row when it takes the monthly REF.
        model_efficiency: Per time resolution, the matrix ``[i, j] = 1 -
            MSE(j, i) / var(j)`` over the versions in ``versions_used``. An
            entry between -10000 and -9999 counts as missing in ``ure`` and is
            left out of the score: it is NaN here.
        versions_used: Percentiles of the versions that took part, per time
            resolution (empty = no REF).
        records_used: Number of records (or periods) the selection used, per
            time resolution.
        tables: Per time resolution a DataFrame with the columns of
            :data:`URE_TABLE_COLUMNS`: the REF version, the USTAR50 version,
            mean, standard error and the 7 percentiles across the 40
            versions. Half-hourly (or hourly) values in umol m-2 s-1 on the
            full-year TIMESTAMP_MIDDLE grid. Daily, weekly and monthly values
            in g C m-2 d-1 and yearly values in g C m-2 y-1, indexed by the
            first day of the period.
        masked: Records left out of all 40 versions (the long gaps of the NEE
            REF selection), on the TIMESTAMP_MIDDLE grid.
        years: Calendar years of the grid.
    """
    percentiles: tuple
    index: dict
    percentile: dict
    scores: pd.DataFrame
    model_efficiency: dict = field(repr=False)
    versions_used: dict = field(repr=False)
    records_used: dict = field(repr=False)
    tables: dict = field(repr=False)
    masked: pd.Series = field(repr=False)
    years: tuple = ()

    def summary(self) -> str:
        """REF percentile per time resolution as text."""
        lines = ["REF u* percentile (ONEFlux ure, model efficiency)", "=" * 50]
        for agg in REF_AGGREGATIONS:
            p = self.percentile.get(agg, np.nan)
            txt = f"{p:g}" if np.isfinite(p) else "none"
            lines.append(f"{agg:<4}{txt:>8}   versions used: {len(self.versions_used[agg]):>2}"
                         f", records used: {self.records_used[agg]}")
        lines.append(f"Records left out: {int(self.masked.sum())} of {len(self.masked)}")
        return "\n".join(lines)


def select_partitioning_reference(values: pd.DataFrame, percentiles=ONEFLUX_REF_PERCENTILES,
                                  *, ust50: pd.Series | None = None,
                                  mask: pd.Series | None = None,
                                  threshold_type: str = 'vut',
                                  ust50_week: pd.Series | None = None,
                                  valid_data_count: int = URE_VALID_DATA_COUNT,
                                  valid_perc_count: int = URE_VALID_PERC_COUNT,
                                  min_7_perc_count: int = URE_MIN_7_PERC_COUNT,
                                  min_3_perc_count: int = URE_MIN_3_PERC_COUNT,
                                  ) -> PartitioningReferenceResult:
    """Select the REF of GPP or RECO as ONEFlux ``ure`` does.

    ONEFlux partitions each of the 40 gap-filled NEE versions (u* percentiles
    1.25 to 98.75) and selects the REF of GPP and of RECO in its step ``ure``,
    for the nighttime and the daytime method (``GPP_NT_VUT_REF``,
    ``RECO_DT_CUT_REF``, ...). The score is the model efficiency of
    :func:`select_ustar_reference`, on the half-hourly, daily, weekly, monthly
    and yearly values. ``ure`` also reports, per record and period, the
    USTAR50 version and the mean, standard error and 7 percentiles across the
    40 versions (``GPP_NT_VUT_MEAN``, ``_SE``, ``_05`` to ``_95``). This
    function returns all of them in :attr:`PartitioningReferenceResult.tables`.

    The code reproduces ``ure`` 1.021 (ure dataset.c), which differs from the
    NEE selection in ``nee_proc``:

    - No quality filter of its own. ``mask`` removes records from the 40
      versions (not from USTAR50). ONEFlux passes the long gaps ``nee_proc``
      left out of its NEE REF selection (``mef_filter`` files, ure
      dataset.c:1478-1541, see :attr:`UstarReferenceResult.masked`). The
      ONEFlux pipeline swaps them: the VUT selection gets the CUT gaps and the
      CUT selection the VUT gaps (``wrappers.py:2196-2199``).
    - Before each selection, periods missing in all versions are dropped, a
      version with fewer than 90 % of the valid values of the most complete
      version is left out, the selection needs at least 10 versions, and then
      every period with a missing value is dropped (ure dataset.c:316-589).
      Otherwise there is no REF.
    - The yearly REF is the monthly REF when one year is left
      (``threshold_type='vut'``, ure dataset.c:1008). ``ure`` does not apply
      this to CUT: a single complete year then gives NaN scores and the first
      version. ONEFlux reports CUT only for 3 or more years. diive also
      selects CUT for fewer years and then applies the rule.
    - Daily values are the mean times ``CO2TOC``, weekly and monthly values
      the mean of the days, yearly values their sum, as in ``nee_proc``.
      Week 52 starts at day 351 plus the number of valid days of the
      USTAR50 version in week 51 (7 when complete), a loop counter ``ure``
      reuses (ure dataset.c:2250). ``ust50_week`` sets that version.
    - The 7 percentiles take the value at position ``ROUND(p / 100 * n)`` of
      the ``n`` sorted valid versions (in float). They need more than 10 valid
      versions, and with fewer than 30 only 25, 50 and 75 are given. The
      standard error is the standard deviation divided by sqrt(40), whatever
      the number of valid versions (ure dataset.c:1052-1083).
    - The statistics are only given at time resolutions with a REF.

    Args:
        values: GPP or RECO of the 40 versions (umol m-2 s-1), one column per
            version in the order of ``percentiles``, NaN = missing. Half-hourly
            or hourly records on a regular DatetimeIndex (TIMESTAMP_MIDDLE, or
            an index named TIMESTAMP_START or TIMESTAMP_END). Each calendar year
            is padded to its full grid with missing values, missing years in
            between are added. Clean the values first as ONEFlux does, with
            :func:`oneflux_ure_cleanup`.
        percentiles: Percentile of each column. Defaults to
            :data:`ONEFLUX_REF_PERCENTILES`.
        ust50: The USTAR50 version (u* percentile 50), same index as
            ``values``. Not a candidate. It gives the ``USTAR50`` column and,
            unless ``ust50_week`` is given, the start of week 52. None counts as
            missing everywhere, as ``ure`` would see it, so week 52 then starts
            at day 351.
        mask: Records to leave out of the 40 versions (True), on any
            DatetimeIndex (e.g. :attr:`UstarReferenceResult.masked`). None
            leaves out none.
        threshold_type: ``'vut'`` or ``'cut'``, for the yearly rule above.
        ust50_week: USTAR50 version that sets the start of week 52. With 3 or
            more years ``ure`` uses the CUT version for both threshold types.
            None uses ``ust50``.
        valid_data_count: ``ure`` option, see :data:`URE_VALID_DATA_COUNT`.
        valid_perc_count: ``ure`` option, see :data:`URE_VALID_PERC_COUNT`.
        min_7_perc_count: ``ure`` option, see :data:`URE_MIN_7_PERC_COUNT`.
        min_3_perc_count: ``ure`` option, see :data:`URE_MIN_3_PERC_COUNT`.

    Returns:
        :class:`PartitioningReferenceResult`.
    """
    from diive.flux.lowres.ustar_detection import _to_middle

    percentiles = tuple(float(p) for p in percentiles)
    if values.shape[1] != len(percentiles):
        raise ValueError(f"values has {values.shape[1]} columns but {len(percentiles)} "
                         f"percentiles were given.")
    if len(percentiles) < 2:
        raise ValueError("The REF selection needs at least two versions.")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("values needs a DatetimeIndex.")
    threshold_type = str(threshold_type).lower()
    if threshold_type not in ('vut', 'cut'):
        raise ValueError(f"threshold_type must be 'vut' or 'cut', got {threshold_type!r}.")
    extra = []
    for name, s in (('ust50', ust50), ('ust50_week', ust50_week)):
        if s is not None and not s.index.equals(values.index):
            raise ValueError(f"{name} needs the index of values.")
        extra.append(np.full(len(values), np.nan) if s is None else s.to_numpy(float))

    k = len(percentiles)
    frame = pd.DataFrame(np.column_stack([values.to_numpy(float), *extra]), index=values.index)
    frame.index.name = values.index.name
    grid, years, rows_per_day = _on_year_grid(frame)
    arr = grid.to_numpy(float).T.copy()
    arr[np.isnan(arr)] = _INVALID
    hh = arr[:k + 1].copy()  # the 40 versions and USTAR50
    week50 = arr[k + 1].copy() if ust50_week is not None else hh[k].copy()

    masked = np.zeros(hh.shape[1], dtype=np.bool_)
    if mask is not None:
        mf = pd.DataFrame({'m': np.asarray(mask, dtype=float)}, index=mask.index)
        mf.index.name = mask.index.name
        masked = _to_middle(mf)['m'].reindex(grid.index).fillna(0.0).to_numpy() > 0
        hh[:k, masked] = _INVALID  # not the 50 % version (ure dataset.c:1520)

    dd = _daily_kernel(hh, rows_per_day, CO2TOC)
    dd_week = _daily_kernel(week50[None, :], rows_per_day, CO2TOC)[0]
    w_starts, w_lengths = _ure_week_windows(years, dd_week)
    matrices = {
        'HH': hh,
        'DD': dd,
        'WW': _window_kernel(dd, w_starts, w_lengths, True),
        'MM': _group_kernel(dd, _month_lengths(years), True),
        'YY': _group_kernel(dd, _year_lengths(years), False),
    }
    days = pd.date_range(pd.Timestamp(years[0], 1, 1), periods=dd.shape[1], freq='D',
                         name='TIMESTAMP_START')
    month_starts = np.concatenate([[0], np.cumsum(_month_lengths(years))[:-1]])
    year_starts = np.concatenate([[0], np.cumsum(_year_lengths(years))[:-1]])
    periods = {'HH': grid.index, 'DD': days, 'WW': days[w_starts],
               'MM': days[month_starts], 'YY': days[year_starts]}

    nq = len(URE_PERCENTILES)
    index_table = np.zeros((k + 1, nq), dtype=np.int64)
    for y in range(1, k + 1):
        for q, p in enumerate(URE_PERCENTILES):
            index_table[y, q] = _ure_percentile_index(p, y)
    keep3 = np.array([p in (25, 50, 75) for p in URE_PERCENTILES])
    yearly_rule = threshold_type == 'vut' or len(years) < 3

    index, percentile, scores, mefs, used, n_used, tables = {}, {}, {}, {}, {}, {}, {}
    for agg in REF_AGGREGATIONS:
        m = matrices[agg]
        sc = np.full(k, np.nan)
        red = _ure_ref_matrix(m[:k].T, valid_data_count, valid_perc_count)
        if red is None:
            idx = -1
            used[agg], n_used[agg] = (), 0
        else:
            reduced, versions = red
            used[agg] = tuple(percentiles[v] for v in versions)
            n_used[agg] = int(reduced.shape[0])
            if agg == 'YY' and yearly_rule and reduced.shape[0] == 1:
                idx = index['MM']  # ure dataset.c:1008
            else:
                i, mess, mef = oneflux_model_efficiency(reduced, fail_on_invalid_score=False)
                idx = int(versions[i])
                sc[versions] = mess
                mefs[agg] = mef
        index[agg] = idx
        percentile[agg] = percentiles[idx] if idx >= 0 else np.nan
        scores[agg] = sc

        n = m.shape[1]
        stats = np.full((n, nq + 2), np.nan)
        if red is not None:
            _ure_stats_kernel(np.ascontiguousarray(m[:k]), index_table, keep3,
                              int(min_3_perc_count), int(min_7_perc_count), stats)
        ref = m[idx].copy() if idx >= 0 else np.full(n, _INVALID)
        u50 = m[k].copy()
        for a in (ref, u50):
            a[(a > -10000.0) & (a <= -9999.0)] = np.nan
        tables[agg] = pd.DataFrame(
            np.column_stack([ref, u50, stats[:, nq], stats[:, nq + 1], stats[:, :nq]]),
            index=periods[agg], columns=list(URE_TABLE_COLUMNS))

    return PartitioningReferenceResult(
        percentiles=percentiles,
        index=index,
        percentile=percentile,
        scores=pd.DataFrame([scores[a] for a in REF_AGGREGATIONS],
                            index=list(REF_AGGREGATIONS), columns=list(percentiles)),
        model_efficiency=mefs,
        versions_used=used,
        records_used=n_used,
        tables=tables,
        masked=pd.Series(masked, index=grid.index, name='REF_MASKED'),
        years=tuple(years),
    )
