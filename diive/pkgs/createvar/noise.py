import numpy as np
import pandas as pd


def add_impulse_noise(
        series: pd.Series,
        factor_low: float = -10,
        factor_high: float = 10,
        contamination: float = 0.04
):
    """Create impulse noise based on series

    kudos:
        https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524

    Args:
        series:
        factor_low:
        factor_high:
        contamination:

    Returns:
        Series with added noise

    """

    # Set parameters
    minimum_noise = factor_low * min(series)
    maximum_noise = factor_high * max(series)
    contamination_noise = int(contamination * len(series))

    # Generate noise sample with values that are higer or lower than a randomly selected values in the original data
    noise_impulse_sample = np.random.default_rng().uniform(minimum_noise, maximum_noise, contamination_noise)

    # Generate an array of zeros with a size that is the difference of the sizes of the original data an the noise sample
    zeros = np.zeros(len(series) - len(noise_impulse_sample))

    # Add noise sample to zeros array to obtain the final noise with the same shape as that of the original data
    noise_impulse = np.concatenate([noise_impulse_sample, zeros])

    # Shuffle the values in the noise to make sure the values are randomly placed.
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
