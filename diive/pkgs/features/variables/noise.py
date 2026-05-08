import numpy as np
import pandas as pd


def generate_noisy_timeseries(
        start_date='2024-01-01',
        periods=365,
        freq='D',
        trend_slope=0.05,
        seasonal_strength=5.0,
        noise_level=2.0,
        outlier_fraction=0.02,
        random_seed=42
):
    """
    Generate synthetic time series DataFrame containing trend, seasonality,
    Gaussian noise, and random outliers.

    The synthetic data is constructed using an additive model:
    y(t) = Trend(t) + Seasonality(t) + Noise(t) + Outliers(t)

    Args:
        start_date (str or datetime): The start date for the time index.
            Defaults to '2024-01-01'.
        periods (int): The number of time steps (data points) to generate.
            Defaults to 365.
        freq (str): The frequency string for the time index (e.g., 'D' for daily,
            'H' for hourly, `30min` for half-hourly). Defaults to 'D'.
        trend_slope (float): The slope of the linear trend component.
            Positive values indicate an upward trend; negative values indicate
            a downward trend. Defaults to 0.05.
        seasonal_strength (float): The amplitude of the sine wave component
            representing seasonality. Defaults to 5.0.
        noise_level (float): The standard deviation of the normal (Gaussian)
            distribution used to generate white noise. Defaults to 2.0.
        outlier_fraction (float): The proportion of data points to replace with
            outliers (between 0.0 and 1.0). Outliers are random spikes with
            magnitude 3x-5x the noise level. Defaults to 0.02 (2%).
        random_seed (int): The seed for the random number generator to ensure
            reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex and the following columns:
            - 'base_signal': The underlying signal (trend + seasonality) without noise.
            - 'noise': The isolated Gaussian noise component.
            - 'observed_value': The final synthetic data (base_signal + noise + outliers).

    """
    np.random.seed(random_seed)

    # Create timestamp index based on the start date and frequency
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Create a time step variable (x-axis) for math calculations
    x = np.linspace(0, 50, periods)

    # Base signal: linear trend + seasonality (sine wave)
    # y = mx + A*sin(Bx)
    trend = trend_slope * np.arange(periods)
    seasonality = seasonal_strength * np.sin(x)
    base_signal = trend + seasonality

    # Add Gaussian (white) noise
    # np.random.normal(mean, std_dev, size)
    noise = np.random.normal(loc=0, scale=noise_level, size=periods)

    # Add outliers: select a random fraction of indices and spike them
    data_with_noise = base_signal + noise

    if outlier_fraction > 0:
        n_outliers = int(periods * outlier_fraction)
        outlier_indices = np.random.choice(periods, size=n_outliers, replace=False)
        # Randomly decide if outlier is positive or negative spike
        signs = np.random.choice([-1, 1], size=n_outliers)
        # Magnitude is 5x to 10x the noise level
        magnitudes = np.random.uniform(5 * noise_level, 10 * noise_level, size=n_outliers)
        data_with_noise[outlier_indices] += signs * magnitudes

    # Assemble DataFrame
    df = pd.DataFrame(data={
        'base_signal': base_signal,
        'noise': noise,
        'observed_value': data_with_noise
    }, index=date_range)

    return df


def add_impulse_noise(
        series: pd.Series,
        factor_low: float = -10,
        factor_high: float = 10,
        contamination: float = 0.04,
        seed: int = None
):
    """Create impulse noise based on series

    kudos:
        https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

    Args:
        series: time series
        factor_low: Low factor for noise creation, defines the lower bound of the noise,
            calculated as factor_low * min(series).
        factor_high: High factor for noise creation, defines the upper bound of the noise,
            calculated as factor_high * max(series).
        contamination: Fraction of *series* to add noise to.
        seed: Random seed.

    Returns:
        Series with added noise

    """

    # Set parameters
    minimum_noise = factor_low * abs(min(series))
    maximum_noise = factor_high * abs(max(series))
    contamination_noise = int(contamination * len(series))

    # Generate noise sample with values that are higer or lower than a randomly selected value in the original data
    noise_impulse_sample = np.random.default_rng(seed=seed).uniform(minimum_noise, maximum_noise, contamination_noise)

    # Generate an array of zeros with a size that is the difference of the sizes of the original data an the noise sample
    zeros = np.zeros(len(series) - len(noise_impulse_sample))

    # Add noise sample to zeros array to obtain the final noise with the same shape as that of the original data
    noise_impulse = np.concatenate([noise_impulse_sample, zeros])

    # Shuffle the values in the noise to make sure the values are randomly placed.
    np.random.seed(seed=seed)
    np.random.shuffle(noise_impulse)

    # Add impulse noise to original data (addition)
    noise_impulse = pd.Series(noise_impulse, index=series.index)
    s_noise = series.add(noise_impulse)

    # import matplotlib.pyplot as plt
    # # s.plot()
    # # plt.show()
    # s_noise.plot()
    # plt.show()

    return s_noise
