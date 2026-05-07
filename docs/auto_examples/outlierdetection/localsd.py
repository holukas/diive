"""
Outlier Detection: Local Standard Deviation Examples
=====================================================

Examples demonstrating the local standard deviation method for outlier detection.
Identifies values that deviate significantly from rolling window statistics.

See: diive.pkgs.outlierdetection.localsd
"""
import warnings

import diive as dv

warnings.filterwarnings('ignore')


def example_localsd_daytime_nighttime():
    """Detect outliers using rolling window with separate day/night thresholds.

    Demonstrates:
    - Adding synthetic impulse noise to clean data
    - Applying separate standard deviation multipliers for day and night
    - Using different rolling window sizes for each period
    - Iterative outlier removal until convergence

    The algorithm identifies values that deviate from a rolling median
    by a specified number of standard deviations. Day/night separation
    is useful when data characteristics vary significantly by time of day.
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

    # Apply local SD with day/night separation
    lsd = dv.LocalSD(
        series=s_noise,
        separate_daytime_nighttime=True,
        n_sd=[3, 2],
        winsize=[48 * 2, 48 * 1],
        constant_sd=False,
        lat=46.0,
        lon=11.0,
        utc_offset=1,
        showplot=True,
        verbose=True
    )
    lsd.calc(repeat=True)

    # Access results
    flag = lsd.get_flag()
    filteredseries = s_noise.copy()
    filteredseries.loc[flag == 2] = None

    print(f"\nOriginal series: {len(s_noise)} records total")
    print(f"Filtered series: {len(filteredseries)} records (same length, outliers set to NaN)")
    print(f"Valid values after filtering: {filteredseries.notna().sum()}")
    print(f"Outliers detected (set to NaN): {(flag == 2).sum()}")
    print(f"\nStatistics:")
    print(f"Original: min={s_noise.min():.2f}, max={s_noise.max():.2f}, mean={s_noise.mean():.2f}")
    print(f"Filtered: min={filteredseries.min():.2f}, max={filteredseries.max():.2f}, mean={filteredseries.mean():.2f}")


def example_localsd_global():
    """Detect outliers using rolling window with single global threshold.

    Demonstrates:
    - Adding synthetic impulse noise to clean data
    - Applying single standard deviation multiplier to all records
    - Using constant (non-rolling) standard deviation for the entire series
    - Iterative outlier removal until convergence

    The algorithm identifies values that deviate from a rolling median
    by a specified number of standard deviations. The global approach
    is simpler and faster when time-of-day variation is not a concern.
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

    # Apply local SD with global threshold
    lsd = dv.LocalSD(
        series=s_noise,
        n_sd=2,
        winsize=48 * 2,
        constant_sd=True,
        showplot=True,
        verbose=True
    )
    lsd.calc(repeat=True)

    # Access results
    flag = lsd.get_flag()
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
    print("Example 1: Local Standard Deviation with Day/Night Separation")
    print("=" * 80)
    example_localsd_daytime_nighttime()

    print("\n" + "=" * 80)
    print("Example 2: Local Standard Deviation with Global Threshold")
    print("=" * 80)
    example_localsd_global()
