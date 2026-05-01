"""
Examples for noise generation and addition using noise module.

Run this script to see noise examples:
    python examples/createvar/noise.py
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import diive as dv


def example_generate_noisy_timeseries_default():
    """Generate synthetic time series with default parameters.

    Demonstrates creating a synthetic time series with trend, seasonality,
    Gaussian noise, and random outliers using default parameters.
    """
    # Generate synthetic data with default parameters
    df = dv.generate_noisy_timeseries()

    print("Generated synthetic time series (daily frequency, 365 days):")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data range: {df['observed_value'].min():.2f} to {df['observed_value'].max():.2f}")

    # Visualize
    fig = plt.figure(facecolor='white', figsize=(16, 5), constrained_layout=True)
    ax1 = fig.add_subplot(111)

    ax1.plot(df.index, df['base_signal'], color='#1565C0', linewidth=2, label='Base signal', alpha=0.7)
    ax1.plot(df.index, df['observed_value'], color='#00BCD4', linewidth=1, label='Observed (with noise & outliers)')
    ax1.scatter(df.index[df['observed_value'].abs() > df['base_signal'].abs() + 10],
                df.loc[df['observed_value'].abs() > df['base_signal'].abs() + 10, 'observed_value'],
                color='#D32F2F', s=50, zorder=5, label='Detected outliers')

    ax1.set_title('Synthetic Time Series: Default Parameters', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.show()


def example_generate_noisy_timeseries_custom():
    """Generate synthetic time series with custom parameters.

    Demonstrates creating synthetic data with custom trend, seasonality,
    noise levels, and outlier fraction for different scenarios.
    """
    # Generate data with stronger trend and seasonality
    df_strong = dv.generate_noisy_timeseries(
        start_date='2023-01-01',
        periods=365,
        freq='D',
        trend_slope=0.2,  # Stronger upward trend
        seasonal_strength=10.0,  # Stronger seasonality
        noise_level=1.0,  # Lower noise
        outlier_fraction=0.01,  # 1% outliers
        random_seed=42
    )

    # Generate data with minimal trend and high noise
    df_noisy = dv.generate_noisy_timeseries(
        start_date='2023-01-01',
        periods=365,
        freq='D',
        trend_slope=0.01,  # Minimal trend
        seasonal_strength=2.0,  # Weak seasonality
        noise_level=5.0,  # High noise
        outlier_fraction=0.05,  # 5% outliers
        random_seed=42
    )

    print("Strong trend & seasonality, low noise:")
    print(f"Mean: {df_strong['observed_value'].mean():.2f}, Std: {df_strong['observed_value'].std():.2f}")
    print(f"Weak trend, high noise:")
    print(f"Mean: {df_noisy['observed_value'].mean():.2f}, Std: {df_noisy['observed_value'].std():.2f}")

    # Visualize comparison
    fig = plt.figure(facecolor='white', figsize=(16, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot 1: Strong signal
    ax1.plot(df_strong.index, df_strong['base_signal'], color='#1565C0', linewidth=2, label='Base signal', alpha=0.7)
    ax1.plot(df_strong.index, df_strong['observed_value'], color='#00BCD4', linewidth=1, label='Observed')
    ax1.set_title('Strong Trend & Seasonality\nLow Noise', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: High noise
    ax2.plot(df_noisy.index, df_noisy['base_signal'], color='#1565C0', linewidth=2, label='Base signal', alpha=0.7)
    ax2.plot(df_noisy.index, df_noisy['observed_value'], color='#D32F2F', linewidth=1, label='Observed')
    ax2.set_title('Weak Trend & Seasonality\nHigh Noise', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.show()


def example_add_impulse_noise():
    """Add impulse noise to an existing time series.

    Demonstrates adding random impulse spikes to a clean time series
    to simulate measurement errors or data corruption.
    """
    # Create a clean synthetic time series
    df = dv.generate_noisy_timeseries(
        periods=365,
        noise_level=0.5,  # Very low noise for clean signal
        outlier_fraction=0.0,  # No outliers
        random_seed=42
    )

    # Add impulse noise
    series_with_impulse = dv.add_impulse_noise(
        series=df['observed_value'],
        factor_low=-5,
        factor_high=5,
        contamination=0.05,  # 5% of data points affected
        seed=42
    )

    print("Added impulse noise to clean time series:")
    print(f"Original max deviation: {(df['observed_value'] - df['observed_value'].mean()).abs().max():.2f}")
    print(f"With impulse noise max deviation: {(series_with_impulse - series_with_impulse.mean()).abs().max():.2f}")
    print(f"Impulse-affected points: ~{int(0.05 * 365)} out of 365")

    # Visualize before and after
    fig = plt.figure(facecolor='white', figsize=(16, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot 1: Clean data
    ax1.plot(df.index, df['observed_value'], color='#1565C0', linewidth=1.5, label='Clean series')
    ax1.plot(df.index, df['base_signal'], color='#00BCD4', linewidth=2, label='Base signal', alpha=0.6)
    ax1.set_title('Before: Clean Time Series', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: With impulse noise
    ax2.plot(df.index, series_with_impulse, color='#D32F2F', linewidth=1.5, label='With impulse noise')
    ax2.plot(df.index, df['base_signal'], color='#00BCD4', linewidth=2, label='Base signal', alpha=0.6)
    ax2.scatter(df.index[df.index.to_period('D').astype(str).isin(
        series_with_impulse.index[
            (series_with_impulse.abs() - df['observed_value'].abs()).abs() > 5
        ].to_period('D').astype(str))],
        series_with_impulse[
            (series_with_impulse.abs() - df['observed_value'].abs()).abs() > 5
        ], color='#FF6F00', s=50, zorder=5, label='Impulse spikes')
    ax2.set_title('After: With Impulse Noise (5% contamination)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.show()


def example_noise_visualization():
    """Visualize components of synthetic time series.

    Shows the individual components (trend, seasonality, noise, outliers)
    and their combination in a multi-panel figure.
    """
    # Generate synthetic data
    df = dv.generate_noisy_timeseries(
        periods=180,  # Half year for clearer visualization
        trend_slope=0.05,
        seasonal_strength=5.0,
        noise_level=2.0,
        outlier_fraction=0.02,
        random_seed=42
    )

    # Create figure with 3 subplots
    fig = plt.figure(facecolor='white', figsize=(16, 9), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Plot 1: Base signal (trend + seasonality)
    ax1.plot(df.index, df['base_signal'], color='#1565C0', linewidth=2, label='Base signal')
    ax1.fill_between(df.index, df['base_signal'], alpha=0.3, color='#1565C0')
    ax1.set_ylabel('Base signal', fontsize=11)
    ax1.set_title('Synthetic Time Series Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Noise component
    ax2.plot(df.index, df['noise'], color='#D32F2F', linewidth=1, label='Gaussian noise')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_ylabel('Noise', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final observed value (signal + noise + outliers)
    ax3.plot(df.index, df['observed_value'], color='#00BCD4', linewidth=1.5, label='Observed (signal + noise + outliers)')
    ax3.plot(df.index, df['base_signal'], color='#1565C0', linewidth=1, linestyle='--', alpha=0.5, label='Underlying signal')
    ax3.set_ylabel('Observed value', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    fig.show()


if __name__ == '__main__':
    print("Running noise generation examples...\n")

    print("1. Generate synthetic time series (default parameters)...")
    example_generate_noisy_timeseries_default()

    print("\n2. Generate with custom parameters...")
    example_generate_noisy_timeseries_custom()

    print("\n3. Add impulse noise to clean series...")
    example_add_impulse_noise()

    print("\n4. Visualization of components...")
    example_noise_visualization()

    print("\nAll examples completed!")
