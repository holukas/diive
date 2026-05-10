import numpy as np
import pandas as pd
from pandas import Series

from diive.pkgs.analysis import gapfinder


def _calculate_gap_sizes(gap_df: pd.DataFrame, series: Series) -> np.ndarray:
    """Calculate gap sizes in records from gap start/end timestamps.

    Args:
        gap_df: DataFrame with GAP_START and GAP_END columns.
        series: Time series to determine median time step.

    Returns:
        Array of gap sizes in records.
    """
    time_delta_seconds = (gap_df['GAP_END'] - gap_df['GAP_START']).dt.total_seconds().astype(int)
    median_step_seconds = series.index.to_series().diff().dt.total_seconds().median() or 1
    gap_sizes = (time_delta_seconds / median_step_seconds).astype(int) + 1
    return gap_sizes.values


def linear_interpolation(series: Series, limit: int = 3, verbose: bool = False) -> Series:
    """Fill gaps in series with linear interpolation up to a specified size.

    Fills missing values in a time series using linear interpolation, with control
    over the maximum consecutive gap size to fill. Gaps larger than the specified
    limit are preserved for other gap-filling methods (ML, MDS).

    Args:
        series: Time series with missing values (NaN).
                Must be a pandas Series with DatetimeIndex.
        limit: Maximum number of consecutive missing values to fill.
               Must be ≥ 1. Default: 3.
               Example: limit=1 fills only isolated single-value gaps,
                       limit=5 fills gaps up to 5 consecutive records.
        verbose: Print summary statistics table showing method parameters,
                 input/output data, gap analysis, and gap size distribution.
                 Default: False.

    Returns:
        pandas.Series: Same index and name as input, with gaps ≤ limit
                      filled using linear interpolation. Gaps exceeding
                      the limit remain as NaN. Original non-gap values
                      are preserved.

    Raises:
        TypeError: If series is not a pandas.Series.
        ValueError: If series is empty, limit < 1, or index is not DatetimeIndex.

    Examples:
        See examples/pkgs/gapfilling/gapfill_interpolate.py for examples demonstrating
        conservative (limit=1) vs. generous (limit=5) gap-filling strategies.
    """
    # Input validation
    if not isinstance(series, Series):
        raise TypeError(f"Expected pandas Series, got {type(series).__name__}")
    if series.empty:
        raise ValueError("Input series cannot be empty")
    if limit < 1:
        raise ValueError(f"Gap size limit must be ≥ 1, got {limit}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    # Input statistics
    n_total = len(series)
    n_missing_input = series.isnull().sum()
    pct_missing_input = 100.0 * n_missing_input / n_total

    # Edge case: no missing values
    if n_missing_input == 0:
        if verbose:
            print(f"\n{'Linear Interpolation Gap-Filling Summary':=^80}")
            print(f"\n{'Parameters':-^80}")
            print(f"  {'Method':<35} Linear Interpolation")
            print(f"  {'Gap size limit':<35} {limit:>4d}  (max consecutive records)")
            print(f"  {'Time series length':<35} {n_total:>9,d}  records")
            print(f"\n{'Input Data':-^80}")
            print(f"  {'Total records':<35} {n_total:>9,d}")
            print(f"  {'Missing values':<35} {n_missing_input:>9,d}  ({pct_missing_input:>6.2f}%)")
            print(f"\n{'Result':-^80}")
            print(f"  {'Status':<35} No gaps found - returning original series")
            print(f"{'':=^80}\n")
        return series.copy()

    # Locate all gaps (without limit filtering to analyze full gap distribution)
    gapfinder_all = gapfinder.GapFinder(series=series, limit=np.inf).get_results()

    # Edge case: no gaps detected
    if len(gapfinder_all) == 0:
        if verbose:
            print(f"\n{'Linear Interpolation Gap-Filling Summary':=^80}")
            print(f"\n{'Parameters':-^80}")
            print(f"  {'Method':<35} Linear Interpolation")
            print(f"  {'Gap size limit':<35} {limit:>4d}  (max consecutive records)")
            print(f"  {'Time series length':<35} {n_total:>9,d}  records")
            print(f"\n{'Input Data':-^80}")
            print(f"  {'Total records':<35} {n_total:>9,d}")
            print(f"  {'Missing values':<35} {n_missing_input:>9,d}  ({pct_missing_input:>6.2f}%)")
            print(f"\n{'Result':-^80}")
            print(f"  {'Status':<35} No gap regions detected - returning original series")
            print(f"{'':=^80}\n")
        return series.copy()

    # Calculate gap sizes for all gaps
    gapfinder_all['gap_size'] = _calculate_gap_sizes(gapfinder_all, series)

    # For interpolation, use only gaps within the limit
    gaps_within_limit = gapfinder_all['gap_size'] <= limit
    gapfinder_agg_df = gapfinder_all[gaps_within_limit].copy()

    # Calculate statistics (efficient - single filter operations)
    gaps_beyond_limit = gapfinder_all[~gaps_within_limit]
    n_gaps_total = len(gapfinder_all)
    n_gaps_filled = len(gapfinder_agg_df)
    n_gaps_skipped = len(gaps_beyond_limit)
    n_records_filled = gapfinder_agg_df['gap_size'].sum() if n_gaps_filled > 0 else 0
    n_records_skipped = gaps_beyond_limit['gap_size'].sum() if n_gaps_skipped > 0 else 0

    # Early exit: no gaps to fill
    if n_gaps_filled == 0:
        if verbose:
            print(f"\n{'Linear Interpolation Gap-Filling Summary':=^80}")
            print(f"\n{'Parameters':-^80}")
            print(f"  {'Method':<35} Linear Interpolation")
            print(f"  {'Gap size limit':<35} {limit:>4d}  (max consecutive records)")
            print(f"  {'Time series length':<35} {n_total:>9,d}  records")
            print(f"\n{'Input Data':-^80}")
            print(f"  {'Total records':<35} {n_total:>9,d}")
            print(f"  {'Missing values':<35} {n_missing_input:>9,d}  ({pct_missing_input:>6.2f}%)")
            print(f"\n{'Gap Analysis (limit={limit})':-^80}")
            print(f"  {'Total gaps detected':<35} {n_gaps_total:>9,d}  (separate regions)")
            print(f"  {'Gaps ≤ '+str(limit)+' record(s)':<35} {n_gaps_filled:>9,d}  (eligible for filling)")
            print(f"  {'Gaps > '+str(limit)+' record(s)':<35} {n_gaps_skipped:>9,d}  (exceed limit)")
            print(f"  {'Values to interpolate':<35} {n_records_filled:>9,d}  (in fillable gaps)")
            print(f"  {'Values skipped':<35} {n_records_skipped:>9,d}  (in too-large gaps)")
            print(f"\n{'Result':-^80}")
            print(f"  {'Status':<35} No fillable gaps - all exceed limit")
            print(f"{'':=^80}\n")
        return series.copy()

    # Interpolate all gaps, then selectively keep only fillable gaps
    series_all_interpolated = series.interpolate(method='linear', limit=None,
                                                  limit_area='inside', limit_direction='both')

    # Vectorized gap-filling: create mask for all fillable gaps
    series_filled = series.copy()
    for _, row in gapfinder_agg_df.iterrows():
        gap_start = row['GAP_START']
        gap_end = row['GAP_END']
        mask = (series.index >= gap_start) & (series.index <= gap_end)
        series_filled.loc[mask] = series_all_interpolated.loc[mask]

    # Output statistics
    n_missing_output = series_filled.isnull().sum()
    pct_missing_output = 100.0 * n_missing_output / n_total
    pct_recovery = 100.0 * (n_missing_input - n_missing_output) / max(n_missing_input, 1)

    if verbose:
        print(f"\n{'Linear Interpolation Gap-Filling Summary':=^80}")

        print(f"\n{'Parameters':-^80}")
        print(f"  {'Method':<35} Linear Interpolation")
        print(f"  {'Gap size limit':<35} {limit:>4d}  (max consecutive records)")
        print(f"  {'Time series length':<35} {n_total:>9,d}  records")

        print(f"\n{'Input Data':-^80}")
        print(f"  {'Total records':<35} {n_total:>9,d}")
        print(f"  {'Missing values':<35} {n_missing_input:>9,d}  ({pct_missing_input:>6.2f}%)")

        print(f"\n{'Gap Analysis (limit={limit})':-^80}")
        print(f"  {'Total gaps detected':<35} {n_gaps_total:>9,d}  (separate regions)")
        print(f"  {'Gaps ≤ '+str(limit)+' record(s)':<35} {n_gaps_filled:>9,d}  (eligible for filling)")
        print(f"  {'Gaps > '+str(limit)+' record(s)':<35} {n_gaps_skipped:>9,d}  (exceed limit)")
        print(f"  {'Values to interpolate':<35} {n_records_filled:>9,d}  (in fillable gaps)")
        print(f"  {'Values skipped':<35} {n_records_skipped:>9,d}  (in too-large gaps)")

        print(f"\n{'Output Data':-^80}")
        print(f"  {'Missing values after':<35} {n_missing_output:>9,d}  ({pct_missing_output:>6.2f}%)")
        n_filled = int(n_missing_input - n_missing_output)
        print(f"  {'Successfully filled':<35} {n_filled:>9,d}  ({pct_recovery:>6.2f}% of {n_missing_input:,d})")

        if n_gaps_total > 0:
            gap_sizes = gapfinder_all['gap_size']
            print(f"\n{'Gap Size Distribution':-^80}")
            print(f"  {'Smallest gap':<35} {gap_sizes.min():>9.0f}  record(s)")
            print(f"  {'Median gap size':<35} {gap_sizes.median():>9.0f}  record(s)")
            print(f"  {'Largest gap':<35} {gap_sizes.max():>9.0f}  record(s)")
            print(f"  {'Mean gap size':<35} {gap_sizes.mean():>9.1f}  record(s)")

        print(f"{'':=^80}\n")

    return series_filled


# See examples/pkgs/gapfilling/gapfill_interpolate.py for usage examples.
