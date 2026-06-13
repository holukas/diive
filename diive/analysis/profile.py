"""
ANALYSIS.PROFILE: DATAFRAME PROFILING
=====================================

Whole-dataset profiling: a compact, per-variable summary table plus a handful
of dataset-level facts (rows, columns, duplicate timestamps, overall coverage,
inferred frequency). Unlike `sstats` (30 metrics for a single series) this is
deliberately lightweight so it can run across every column of a wide dataframe
at once — the kind of "data profiling" overview you want right after loading.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from pandas.api import types as ptypes

#: Column order of the per-variable profile table returned by `profile_dataframe`.
PROFILE_COLUMNS = [
    'VARIABLE', 'DTYPE', 'COUNT', 'MISSING', 'MISSING_PERC', 'N_GAPS',
    'N_UNIQUE', 'N_ZEROS', 'CONSTANT', 'MEAN', 'SD', 'MIN', 'MEDIAN', 'MAX',
]


def count_gaps(s: pd.Series) -> int:
    """Number of gaps = maximal runs of consecutive missing values.

    A single missing value is one gap; ten consecutive missing values are still
    one gap. Counts the *starts* of NaN runs (cheap, no grouping needed) — the
    same notion of "gap" used by `GapStats`/`GapFinder`.
    """
    isna = s.isna()
    if not isna.any():
        return 0
    # A run starts where the value is NaN but the previous one was not.
    return int((isna & ~isna.shift(fill_value=False)).sum())


def profile_dataframe(df: DataFrame) -> DataFrame:
    """Per-variable profiling table — one row per column.

    Columns (see `PROFILE_COLUMNS`): VARIABLE, DTYPE, COUNT (non-missing),
    MISSING, MISSING_PERC, N_GAPS (consecutive-NaN runs), N_UNIQUE, N_ZEROS,
    CONSTANT (≤1 unique non-missing value), and the numeric summaries MEAN, SD,
    MIN, MEDIAN, MAX (NaN for non-numeric columns).

    Args:
        df: Dataframe to profile (any index; numeric stats apply to numeric cols).

    Returns:
        DataFrame with one row per input column and the columns above. Empty
        (but correctly-columned) when `df` has no columns.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        count = int(s.notna().sum())
        missing = n - count
        is_numeric = ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s)
        numeric = s if is_numeric else None
        rows.append({
            'VARIABLE': str(col),
            'DTYPE': str(s.dtype),
            'COUNT': count,
            'MISSING': missing,
            'MISSING_PERC': round(100.0 * missing / n, 2) if n else 0.0,
            'N_GAPS': count_gaps(s),
            'N_UNIQUE': int(s.nunique(dropna=True)),
            'N_ZEROS': int((numeric == 0).sum()) if numeric is not None else 0,
            'CONSTANT': bool(s.nunique(dropna=True) <= 1),
            'MEAN': float(numeric.mean()) if numeric is not None and count else float('nan'),
            'SD': float(numeric.std()) if numeric is not None and count else float('nan'),
            'MIN': float(numeric.min()) if numeric is not None and count else float('nan'),
            'MEDIAN': float(numeric.median()) if numeric is not None and count else float('nan'),
            'MAX': float(numeric.max()) if numeric is not None and count else float('nan'),
        })
    return DataFrame(rows, columns=PROFILE_COLUMNS)


def dataframe_overview(df: DataFrame) -> dict:
    """Dataset-level facts for a profiling header.

    Returns a dict with: n_rows, n_cols, n_cells, missing_cells, missing_perc,
    duplicate_timestamps (duplicated index labels), duplicate_rows, start, end,
    freq (the index `freqstr` if set, else `pd.infer_freq`, else None), and
    memory_bytes (deep memory usage). Timestamp/freq fields are None when the
    index is not a DatetimeIndex.
    """
    n_rows, n_cols = df.shape
    n_cells = n_rows * n_cols
    missing_cells = int(df.isna().to_numpy().sum()) if n_cells else 0

    is_dt = isinstance(df.index, pd.DatetimeIndex)
    freq = None
    if is_dt:
        freq = df.index.freqstr
        if freq is None and n_rows > 2:
            try:
                freq = pd.infer_freq(df.index)
            except (ValueError, TypeError):
                freq = None

    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_cells': n_cells,
        'missing_cells': missing_cells,
        'missing_perc': round(100.0 * missing_cells / n_cells, 2) if n_cells else 0.0,
        'duplicate_timestamps': int(df.index.duplicated().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'start': df.index.min() if is_dt and n_rows else None,
        'end': df.index.max() if is_dt and n_rows else None,
        'freq': freq,
        'memory_bytes': int(df.memory_usage(deep=True).sum()),
    }
