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
    if is_invalid(s):
        return -1
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
def _group_kernel(dd, lengths, divide):
    """Weekly or monthly means (``divide``) or yearly sums of the daily values."""
    k = dd.shape[0]
    ng = lengths.shape[0]
    out = np.empty((k, ng))
    for p in range(k):
        pos = 0
        for g in range(ng):
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
            pos += z
    return out


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


def oneflux_model_efficiency(values):
    """Model efficiency scores and REF index as ``get_reference`` (dataset.c:982-1140).

    Args:
        values: 2-D array, one row per record (or aggregated period) and one
            column per version, NaN or -9999 = missing.

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
    from diive.flux.lowres.ustar_detection import _full_year, _record_step, _to_middle

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

    cols = list(range(len(percentiles)))
    frame = pd.DataFrame(np.column_stack([nee.to_numpy(float), qc.to_numpy(float)]),
                         index=nee.index)
    frame.index.name = nee.index.name
    frame = _to_middle(frame).sort_index()
    step = _record_step(frame.index)
    rows_per_day = int(round(pd.Timedelta('1D') / step))
    if rows_per_day not in (24, 48):
        raise ValueError(f"Records must be half-hourly or hourly, got a step of {step}.")
    years = list(range(int(frame.index.year.min()), int(frame.index.year.max()) + 1))
    grid = pd.concat([_full_year(frame, y, step) for y in years])

    k = len(cols)
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
