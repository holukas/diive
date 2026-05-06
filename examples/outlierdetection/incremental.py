"""
Outlier Detection: Z-Score Increments Examples
===============================================

Examples demonstrating the z-score increments method for outlier detection.
Identifies outliers based on abrupt changes between consecutive values.

See: diive.pkgs.outlierdetection.incremental
"""
import warnings

import diive as dv

warnings.filterwarnings('ignore')


def example_incremental_zscore():
    """Detect outliers using z-score of record increments.

    Demonstrates:
    - Adding synthetic impulse noise to clean data
    - Detecting outliers based on abrupt changes between records
    - Z-score calculation on forward, backward, and combined increments
    - Iterative outlier removal until convergence

    The algorithm calculates three types of increments for each value:
    - Forward increment: difference from previous value
    - Backward increment: difference to next value
    - Combined increment: sum of forward and backward
    Values flagged as outliers in all three are removed.
    """
    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()

    # Add synthetic impulse noise (spikes)
    s_noise = dv.add_impulse_noise(
        series=s,
        factor_low=-10,
        factor_high=3,
        contamination=0.04,
        seed=42
    )
    s_noise.name = f"{s.name}+noise"

    # Apply z-score increments outlier detection
    zsi = dv.zScoreIncrements(
        series=s_noise,
        thres_zscore=3,
        showplot=True,
        verbose=True
    )
    zsi.calc(repeat=True)

    # Access results
    flag = zsi.get_flag()
    filteredseries = s_noise.copy()
    filteredseries.loc[flag == 2] = None

    print(f"\nOriginal series: {len(s_noise)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers detected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics:")
    print(f"Original: min={s_noise.min():.2f}, max={s_noise.max():.2f}, mean={s_noise.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")


if __name__ == '__main__':
    print("=" * 80)
    print("Example: Z-Score Increments Outlier Detection")
    print("=" * 80)
    example_incremental_zscore()
