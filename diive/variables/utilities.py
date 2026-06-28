"""
UTILITIES: TESTING AND SYNTHETIC DATA GENERATION
=================================================

Generate synthetic time series with noise and add impulse disturbances for testing.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd

#: Arithmetic methods accepted by :func:`combine_variables`, mapping each to the
#: pandas Series operator and the operation's identity (the fill value used for a
#: missing operand when ``keep_overlap_only=False``).
_COMBINE_METHODS = {
    'add': ('add', 0),
    'subtract': ('sub', 0),
    'multiply': ('mul', 1),
    'divide': ('truediv', 1),
}


def combine_variables(
        series1: pd.Series,
        series2: pd.Series,
        method: str = 'multiply',
        keep_overlap_only: bool = True,
        name: str | None = None,
) -> pd.Series:
    """Combine two time series element-wise by an arithmetic operation.

    The two series are aligned on their (timestamp) index and combined record by
    record. ``method`` selects the operation:

    =============  ========  ============================
    method         symbol    identity (fill value)
    =============  ========  ============================
    ``add``           +              0
    ``subtract``      -              0
    ``multiply``      *              1
    ``divide``        /              1
    =============  ========  ============================

    ``method='fillgaps'`` is a special case: it keeps ``series1`` and fills only
    its gaps (NaNs) with the matching values of ``series2`` (a
    ``series1.combine_first(series2)``). ``keep_overlap_only`` is ignored for it
    (filling gaps is, by definition, a union).

    Args:
        series1: First (left-hand) series.
        series2: Second (right-hand) series.
        method: One of ``'add'``, ``'subtract'``, ``'multiply'``, ``'divide'``,
            ``'fillgaps'``.
        keep_overlap_only: If True (default), the result is missing (NaN) wherever
            *either* input is missing — only overlapping (both-present) records get
            a value. If False, a missing value is treated as the operation's
            identity (0 for add/subtract, 1 for multiply/divide), so records
            present in only one input survive. Ignored for ``'fillgaps'``.
        name: Name for the returned series. Defaults to a generated
            ``{series1.name}_{METHOD}_{series2.name}`` label.

    Returns:
        The combined series, indexed by the union of the two inputs' indexes.
    """
    if method == 'fillgaps':
        # Keep series1, fill only its gaps with series2 (union over the indexes).
        result = series1.combine_first(series2)
    elif method in _COMBINE_METHODS:
        op, identity = _COMBINE_METHODS[method]
        # fill_value=None keeps pandas' default (NaN where either operand is NaN);
        # the identity lets a one-sided record survive when overlap is not required.
        fill_value = None if keep_overlap_only else identity
        result = getattr(series1, op)(series2, fill_value=fill_value)
    else:
        choices = sorted(list(_COMBINE_METHODS) + ['fillgaps'])
        raise ValueError(f"Unknown method '{method}'. Choose from {choices}.")
    if name is None:
        name = f"{series1.name}_{method.upper()}_{series2.name}"
    result.name = name
    return result


def combine_variables_to_code(
        var1: str,
        var2: str,
        method: str = 'multiply',
        keep_overlap_only: bool = True,
        name: str | None = None,
) -> str:
    """Render a :func:`combine_variables` call as a runnable snippet.

    Assumes ``dv`` and a DataFrame ``df`` are already in scope. Returns the
    snippet ending in a newline (the result is bound to ``combined``).

    Args:
        var1: Column name for the first (left-hand) operand.
        var2: Column name for the second (right-hand) operand.
        method: Combine method (see :func:`combine_variables`).
        keep_overlap_only: Rendered only when False (True is the default).
        name: Output series name, rendered as ``name=`` when given.
    """
    args = [f"df[{var1!r}]", f"df[{var2!r}]", f"method={method!r}"]
    # keep_overlap_only is meaningless for the gap-fill method (always a union).
    if not keep_overlap_only and method != 'fillgaps':
        args.append("keep_overlap_only=False")
    if name:
        args.append(f"name={name!r}")
    return f"combined = dv.variables.combine_variables({', '.join(args)})\n"


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

    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    x = np.linspace(0, 50, periods)

    trend = trend_slope * np.arange(periods)
    seasonality = seasonal_strength * np.sin(x)
    base_signal = trend + seasonality

    noise = np.random.normal(loc=0, scale=noise_level, size=periods)

    data_with_noise = base_signal + noise

    if outlier_fraction > 0:
        n_outliers = int(periods * outlier_fraction)
        outlier_indices = np.random.choice(periods, size=n_outliers, replace=False)
        signs = np.random.choice([-1, 1], size=n_outliers)
        magnitudes = np.random.uniform(5 * noise_level, 10 * noise_level, size=n_outliers)
        data_with_noise[outlier_indices] += signs * magnitudes

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

    minimum_noise = factor_low * abs(min(series))
    maximum_noise = factor_high * abs(max(series))
    contamination_noise = int(contamination * len(series))

    noise_impulse_sample = np.random.default_rng(seed=seed).uniform(minimum_noise, maximum_noise, contamination_noise)

    zeros = np.zeros(len(series) - len(noise_impulse_sample))

    noise_impulse = np.concatenate([noise_impulse_sample, zeros])

    np.random.seed(seed=seed)
    np.random.shuffle(noise_impulse)

    noise_impulse = pd.Series(noise_impulse, index=series.index)
    s_noise = series.add(noise_impulse)

    return s_noise
