"""
Examples for MaxCovariance lag detection using time series covariance analysis.

The time lag between two variables (e.g., wind and scalar concentration) can be
determined by finding the maximum covariance at different lag times. This is useful
for detecting inlet tube delays and other measurement timing issues in eddy covariance data.

Run this script to see MaxCovariance examples:
    python examples/echires/lag.py

See Also
--------
diive.MaxCovariance : Time lag detection class documentation.
"""
import pandas as pd
import numpy as np

import diive as dv


def example_lag_synthetic():
    """Detect time lag between variables using synthetic data.

    Demonstrates MaxCovariance class by:
    1. Creating synthetic turbulent wind (w) and scalar (c) data
    2. Adding a known time lag to the scalar (inlet tube delay)
    3. Using MaxCovariance to recover the lag
    4. Plotting the covariance function to visualize the peak

    This example shows how MaxCovariance automatically detects the lag
    between two variables by finding the maximum covariance peak.
    """

    print("=" * 80)
    print("Example: Time Lag Detection Using Synthetic Data")
    print("=" * 80)

    # Create synthetic high-resolution data (10 Hz, 30 minutes = 18000 records)
    n_records = 18000
    np.random.seed(42)

    # Wind components (m/s) - smooth turbulent structures
    window_size = 20  # Smooth over ~2 seconds at 10 Hz
    w_noise = np.random.normal(0.0, 0.2, n_records)
    w = np.convolve(w_noise, np.ones(window_size)/window_size, mode='same')

    # Scalar (ppb) - strongly correlated with lagged w
    # Introduce a known lag: 1 second = 10 records at 10 Hz
    lag_records = 10  # 1 second lag
    w_lagged = np.roll(w, lag_records)
    w_lagged[:lag_records] = w_lagged[lag_records]  # Fill initial lag

    c_noise = np.random.normal(0, 1.5, n_records)
    c_smooth = np.convolve(c_noise, np.ones(window_size)/window_size, mode='same')
    c = 300 + c_smooth + 2.5 * w_lagged  # Strong correlation with lagged wind

    # Create turbulent fluctuations (deviations from mean)
    w_turb = w - np.mean(w)
    c_turb = c - np.mean(c)

    # Create DataFrame
    df = pd.DataFrame({
        'w_TURB': w_turb,
        'c_TURB': c_turb
    })

    print(f"\nData shape: {df.shape}")
    print(f"Wind (w) statistics:")
    print(f"  Mean: {w_turb.mean():.6f} m/s")
    print(f"  Std:  {w_turb.std():.6f} m/s")
    print(f"Scalar (c) statistics:")
    print(f"  Mean: {c_turb.mean():.6f} ppb")
    print(f"  Std:  {c_turb.std():.6f} ppb")
    print(f"\nKnown lag in synthetic data: {lag_records} records (1.0 second at 10 Hz)")

    # Detect time lag using MaxCovariance
    mc = dv.MaxCovariance(
        df=df,
        var_reference='w_TURB',
        var_lagged='c_TURB',
        lgs_winsize_from=-100,  # records
        lgs_winsize_to=100,      # records
        shift_stepsize=1,        # records per step
        segment_name='synthetic_data'
    )

    mc.run()
    cov_df, props_peak_auto = mc.get()

    print(f"\n" + "=" * 80)
    print("Lag Detection Results")
    print("=" * 80)

    # Get results
    if mc.idx_peak_cov_abs_max is not False:
        detected_lag = int(cov_df.iloc[mc.idx_peak_cov_abs_max]['shift'])
        max_cov = cov_df.iloc[mc.idx_peak_cov_abs_max]['cov']
        print(f"\n[OK] Maximum covariance detected at lag: {detected_lag} records")
        print(f"  Covariance value: {max_cov:.6f}")
        print(f"  Expected lag: {lag_records} records")
        print(f"  Error: {abs(detected_lag - lag_records)} records")
    else:
        print("\n[ERROR] No peak detected")

    if mc.idx_peak_auto is not False:
        auto_lag = int(cov_df.iloc[mc.idx_peak_auto]['shift'])
        print(f"\n[OK] Auto-detected peak at lag: {auto_lag} records")
        print(f"  Peak score: {props_peak_auto['peak_score']:.2f}")
        print(f"  Peak rank: {props_peak_auto['peak_rank']:.0f}")
    else:
        print("\n[INFO] No auto-detected peak")

    # Display covariance plot
    print(f"\n" + "=" * 80)
    print("Covariance Function Visualization")
    print("=" * 80)

    try:
        fig = mc.plot_scatter_cov(
            title='Covariance vs Time Lag (Synthetic Data)',
            showplot=True
        )
        print("[OK] Covariance plot generated successfully")
    except Exception as e:
        print(f"Plot display info: {type(e).__name__}")


if __name__ == '__main__':
    print("=" * 80)
    print("MaxCovariance Examples: Time Lag Detection")
    print("=" * 80)
    print()

    example_lag_synthetic()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
