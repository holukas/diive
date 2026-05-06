"""
Outlier Detection: Absolute Limits Examples
=============================================

Examples demonstrating absolute value limits for outlier detection,
with separate daytime/nighttime thresholds.

See: diive.pkgs.outlierdetection.absolutelimits
"""
import numpy as np
import pandas as pd

import diive as dv


def example_absolute_limits_basic():
    """Simple absolute limits filter on random data.

    Demonstrates:
    - Creating a time series with random values
    - Applying min/max threshold constraints
    - Accessing filtered series and quality flags
    """
    np.random.seed(100)
    rows = 1000
    data = np.random.rand(rows) * 100  # Random numbers between 0 and 100
    tidx = pd.date_range('2019-01-01 00:30:00', periods=rows, freq='30min')
    series = pd.Series(data, index=tidx, name='TESTDATA')

    # Apply absolute limits filter (reject values < 6 or > 74)
    al = dv.AbsoluteLimits(series=series, minval=6, maxval=74, idstr='99', showplot=True, verbose=True)
    al.calc()

    # Access results
    filteredseries = al.filteredseries
    flag = al.flag

    print(f"\nOriginal series: {len(series)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers rejected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics comparison:")
    print(f"Original: min={series.min():.2f}, max={series.max():.2f}, mean={series.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")


def example_absolute_limits_daytime_nighttime():
    """Absolute limits with separate daytime and nighttime thresholds.

    Demonstrates:
    - Different acceptance ranges for day vs night (common in flux data)
    - Latitude/longitude-based daytime detection
    - Quality flagging with daytime/nighttime visualization

    Use case: CO2 flux filtering where daytime has photosynthesis signal
    (wider range) while nighttime respiration is narrower and more stable.
    """
    np.random.seed(100)
    rows = 1000
    data = np.random.rand(rows) * 100  # Random numbers between 0 and 100
    tidx = pd.date_range('2019-01-01 00:30:00', periods=rows, freq='30min')
    series = pd.Series(data, index=tidx, name='NEE')

    # Apply different thresholds for day vs night
    # Daytime: wider range [6.2, 74.9] (photosynthesis variability)
    # Nighttime: narrower range [29.5, 47.4] (stable respiration)
    al = dv.AbsoluteLimitsDaytimeNighttime(
        series=series,
        daytime_minmax=[6.2, 74.9],
        nighttime_minmax=[29.5, 47.4],
        idstr='99',
        lat=47.286417,  # Swiss FluxNet site
        lon=7.733750,
        utc_offset=1,
        showplot=True,
        verbose=True
    )
    al.calc()

    # Access results
    filteredseries = al.filteredseries
    flag = al.flag

    print(f"\nOriginal series: {len(series)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers rejected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics:")
    print(f"Original: min={series.min():.2f}, max={series.max():.2f}, mean={series.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")

    # Show daytime vs nighttime rejection counts
    daytime_outliers = ((flag == 2) & al.is_daytime).sum()
    nighttime_outliers = ((flag == 2) & al.is_nighttime).sum()
    print(f"\nOutliers by time of day:")
    print(
        f"  Daytime: {daytime_outliers} outliers ({daytime_outliers / al.is_daytime.sum() * 100:.1f}% of daytime records)")
    print(
        f"  Nighttime: {nighttime_outliers} outliers ({nighttime_outliers / al.is_nighttime.sum() * 100:.1f}% of nighttime records)")


if __name__ == '__main__':
    print("=" * 80)
    print("Example 1: Basic Absolute Limits")
    print("=" * 80)
    example_absolute_limits_basic()

    print("\n" + "=" * 80)
    print("Example 2: Daytime/Nighttime Absolute Limits")
    print("=" * 80)
    example_absolute_limits_daytime_nighttime()
