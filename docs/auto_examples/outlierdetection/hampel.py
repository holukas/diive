"""
Outlier Detection: Hampel Filter Examples
==========================================

Examples demonstrating the Hampel filter (Median Absolute Deviation) for
robust outlier detection with optional daytime/nighttime separation.

See: diive.pkgs.preprocessing.outlierdetection.hampel
"""
import warnings

import diive as dv

warnings.filterwarnings('ignore')


def example_hampel_with_impulse_noise():
    """Hampel filter on data with impulse noise (spikes).

    Demonstrates:
    - Adding synthetic impulse noise to clean data
    - Applying Hampel filter with separate day/night thresholds
    - Visualization of detected outliers
    - Iterative outlier removal until convergence

    The Hampel filter uses Median Absolute Deviation (MAD) to detect values
    that deviate significantly from local trend, making it robust to spikes
    and outliers while preserving underlying patterns.
    """
    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()

    # Add synthetic impulse noise (spikes)
    s_noise = dv.add_impulse_noise(
        series=s,
        factor_low=-10,
        factor_high=4,
        contamination=0.04,
        seed=42
    )
    s_noise.name = f"{s.name}+noise"

    # Apply Hampel filter with day/night separation
    ham = dv.Hampel(
        series=s_noise,
        n_sigma=5.5,  # Threshold for outlier detection
        window_length=48 * 13,  # 13 days of half-hourly data
        use_differencing=True,  # Apply to double-differenced data
        separate_day_night=True,
        showplot=True,
        verbose=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )
    ham.calc(repeat=False)

    # Access results
    filteredseries = ham.filteredseries
    flag = ham.flag

    print(f"\nOriginal series: {len(s_noise)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers detected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics:")
    print(f"Original: min={s_noise.min():.2f}, max={s_noise.max():.2f}, mean={s_noise.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")


def example_hampel_global_threshold():
    """Hampel filter with single global threshold for all data.

    Demonstrates:
    - Applying consistent threshold across entire time series
    - No day/night separation (simpler, faster)
    - Useful when time-of-day variation is not a concern
    - Window-based robust outlier detection using MAD

    The algorithm detects local outliers by comparing each value to its
    neighborhood (rolling window) median and median absolute deviation.
    """
    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()

    # Add synthetic impulse noise (spikes)
    s_noise = dv.add_impulse_noise(
        series=s,
        factor_low=-10,
        factor_high=4,
        contamination=0.04,
        seed=42
    )
    s_noise.name = f"{s.name}+noise"

    # Apply Hampel filter with global threshold (no day/night separation)
    ham = dv.Hampel(
        series=s_noise,
        n_sigma=5.5,  # Global threshold applied to all records
        window_length=48 * 13,  # 13-day rolling window
        use_differencing=True,
        separate_day_night=False,  # Single threshold for all
        showplot=True,
        verbose=True
    )
    ham.calc(repeat=True)  # Iterate until all outliers removed

    # Access results
    filteredseries = ham.filteredseries
    flag = ham.flag

    print(f"\nOriginal series: {len(s_noise)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers detected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics:")
    print(f"Original: min={s_noise.min():.2f}, max={s_noise.max():.2f}, mean={s_noise.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")


if __name__ == '__main__':
    print("=" * 80)
    print("Example 1: Hampel Filter with Impulse Noise (Day/Night Separation)")
    print("=" * 80)
    example_hampel_with_impulse_noise()

    print("\n" + "=" * 80)
    print("Example 2: Hampel Filter with Global Threshold")
    print("=" * 80)
    example_hampel_global_threshold()
