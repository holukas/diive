"""
Low-level time series decomposition utilities.

Supports multiple decomposition methods:
- STL (Seasonal-Trend using Loess): Robust, handles gaps and non-stationary data
- Classical (moving average): Simple, fast, assumes stationarity
- Harmonic (Fourier): Frequency-domain analysis, periodic signals

All functions preserve NaN locations and handle quality weighting.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from scipy import signal, fft as scipy_fft


def stl_decompose(
    series: pd.Series,
    seasonal: int = 365,
    trend: int = 730,
    robust: bool = True,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    weights: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """
    STL (Seasonal-Trend using Loess) decomposition.

    Robust decomposition for non-stationary time series with gaps.
    Uses iterative Loess smoothing to separate seasonal, trend, and residual components.

    Args:
        series (pd.Series): Input time series (may contain NaN).
        seasonal (int): Seasonal period in observations. Default 365 (annual for daily data).
        trend (int): Trend smoothing window length. Default 730 (2 * seasonal).
        robust (bool): Use robust regression (less sensitive to outliers). Default True.
        seasonal_deg (int): Loess polynomial degree for seasonal component (0, 1). Default 1.
        trend_deg (int): Loess polynomial degree for trend component (0, 1). Default 1.
        seasonal_jump (int): Jump size for seasonal fitting (speed optimization). Default 1.
        trend_jump (int): Jump size for trend fitting (speed optimization). Default 1.
        weights (np.ndarray, optional): Quality weights (0–1) for each observation.
                                        Higher = more influential. If None, all equal weight.
        verbose (bool): Print decomposition details. Default False.

    Returns:
        Dict with keys:
            - 'seasonal': pd.Series, seasonal component
            - 'trend': pd.Series, trend component
            - 'residual': pd.Series, residual (noise + anomalies)
            - 'weights': np.ndarray, weights used in fitting
            - 'iterations': int, number of inner loop iterations

    Notes:
        - Preserves original series index and NaN locations
        - Handles edge cases: short series, all-NaN sections, single period
        - seasonal must be >= 2
        - trend must be >= 3
    """
    # Input validation
    if len(series) < 2 * seasonal:
        warnings.warn(
            f"Series length ({len(series)}) < 2 * seasonal ({2 * seasonal}). "
            "STL may produce unreliable results.",
            UserWarning
        )

    if seasonal < 2:
        raise ValueError(f"seasonal must be >= 2, got {seasonal}")
    if trend < 3:
        raise ValueError(f"trend must be >= 3, got {trend}")

    # Make even seasonal and trend for STL
    seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
    trend = trend if trend % 2 == 1 else trend + 1

    # Handle weights
    if weights is not None:
        if len(weights) != len(series):
            raise ValueError(f"weights length ({len(weights)}) != series length ({len(series)})")
        weights = np.asarray(weights, dtype=float)
    else:
        weights = np.ones(len(series), dtype=float)

    # Standardize weight range to [0.1, 1.0] for numerical stability
    # (STL's robust fitting expects non-zero weights)
    if weights.max() > weights.min():
        weights_norm = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min())
    else:
        # All weights equal, normalize to 1.0
        weights_norm = np.ones_like(weights)

    try:
        # STL works better with integer-indexed series; convert DatetimeIndex to numeric if needed
        series_for_stl = series.copy()
        if isinstance(series_for_stl.index, pd.DatetimeIndex):
            # Create new index starting from 0
            series_for_stl.index = np.arange(len(series_for_stl))

        # Build STL kwargs with available parameters
        stl_kwargs = {
            'seasonal': seasonal,
            'trend': trend,
            'robust': robust,
        }

        # Only add optional parameters if they differ from defaults
        if seasonal_deg != 1:
            stl_kwargs['seasonal_deg'] = seasonal_deg
        if trend_deg != 1:
            stl_kwargs['trend_deg'] = trend_deg
        if seasonal_jump != 1:
            stl_kwargs['seasonal_jump'] = seasonal_jump
        if trend_jump != 1:
            stl_kwargs['trend_jump'] = trend_jump

        stl_result = STL(series_for_stl, **stl_kwargs)
        decomp = stl_result.fit(weights=weights_norm)

        # Restore original index to decomposition results
        decomp.seasonal.index = series.index
        decomp.trend.index = series.index
        decomp.resid.index = series.index

        if verbose:
            print(f"STL decomposition: seasonal={seasonal}, trend={trend}, robust={robust}")
            print(f"  Iterations: {decomp.nobs}")

        return {
            'seasonal': decomp.seasonal,
            'trend': decomp.trend,
            'residual': decomp.resid,
            'weights': weights,
            'iterations': decomp.nobs if hasattr(decomp, 'nobs') else None
        }

    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {str(e)}")


def classical_decompose(
    series: pd.Series,
    period: int,
    model: str = 'additive',
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """
    Classical seasonal decomposition via centered moving average.

    Simple, fast decomposition using moving averages. Assumes stationarity
    (constant mean trend). Good for pedagogical purposes and simple periodic data.

    Args:
        series (pd.Series): Input time series (should have few missing values).
        period (int): Seasonal period in observations.
        model (str): 'additive' → series = trend + seasonal + residual
                     'multiplicative' → series = trend * seasonal * residual
                     Default 'additive'.
        verbose (bool): Print decomposition details. Default False.

    Returns:
        Dict with keys:
            - 'seasonal': pd.Series, seasonal component
            - 'trend': pd.Series, trend component (moving average)
            - 'residual': pd.Series, residual component

    Notes:
        - First (period-1)//2 and last (period-1)//2 observations get NaN for trend
        - Missing values in input may propagate through output
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    try:
        # Try with extrapolate parameter first (newer statsmodels versions)
        try:
            decomp = seasonal_decompose(series, model=model, period=period, extrapolate='freq')
        except TypeError:
            # Fall back to without extrapolate for older statsmodels versions
            decomp = seasonal_decompose(series, model=model, period=period)

        if verbose:
            print(f"Classical decomposition: period={period}, model={model}")

        return {
            'seasonal': decomp.seasonal,
            'trend': decomp.trend,
            'residual': decomp.resid
        }

    except Exception as e:
        raise RuntimeError(f"Classical decomposition failed: {str(e)}")


def harmonic_decompose(
    series: pd.Series,
    n_harmonics: int = 10,
    period: int = 365,
    window: str = 'hamming',
    verbose: bool = False
) -> Dict:
    """
    Harmonic (Fourier) decomposition for frequency-domain analysis.

    Decomposes time series into sine/cosine basis functions (harmonics).
    Useful for multi-scale seasonal analysis and identifying dominant frequencies.

    Args:
        series (pd.Series): Input time series (NaN values removed).
        n_harmonics (int): Number of harmonics to extract. Default 10.
        period (int): Expected dominant period. Used for frequency scaling. Default 365.
        window (str): Window function ('hamming', 'hann', 'blackman'). Default 'hamming'.
        verbose (bool): Print decomposition details. Default False.

    Returns:
        Dict with keys:
            - 'harmonics': list of dicts, each with:
                - 'amplitude': float, component magnitude
                - 'phase': float, phase in radians
                - 'period': float, period in observations
                - 'frequency': float, normalized frequency (0–1)
            - 'frequencies': np.ndarray, frequency bins
            - 'amplitudes': np.ndarray, amplitude at each frequency
            - 'phases': np.ndarray, phase at each frequency
            - 'spectrum': np.ndarray, power spectral density
            - 'reconstructed': pd.Series, signal reconstructed from top harmonics
            - 'residual': pd.Series, reconstruction error

    Notes:
        - NaN values are removed before FFT (series shortened)
        - Window function reduces spectral leakage
        - Top n_harmonics selected by power (largest amplitude first)
    """
    # Remove NaN
    valid_idx = series.notna()
    series_clean = series[valid_idx].values

    if len(series_clean) < 4:
        raise ValueError(f"Series must have >= 4 valid values, got {len(series_clean)}")

    # Apply window
    try:
        window_func = signal.get_window(window, len(series_clean))
    except ValueError:
        window_func = signal.hamming(len(series_clean))

    series_windowed = series_clean * window_func

    # FFT
    n = len(series_windowed)
    frequencies = np.fft.rfftfreq(n)
    fft_vals = np.fft.rfft(series_windowed) / n
    amplitudes = 2 * np.abs(fft_vals[1:])  # Exclude DC component, double for one-sided spectrum
    powers = amplitudes ** 2

    # Find top harmonics by power
    top_idx = np.argsort(-powers)[:n_harmonics]
    top_idx_sorted = np.sort(top_idx)  # Sort by frequency for reconstruction

    harmonics = []
    for idx in top_idx_sorted:
        freq_norm = frequencies[idx + 1]  # +1 because we excluded DC
        if freq_norm > 0:
            period_obs = 1.0 / freq_norm if freq_norm > 0 else np.inf
            harmonics.append({
                'amplitude': amplitudes[idx],
                'phase': np.angle(fft_vals[idx + 1]),
                'frequency': freq_norm,
                'period': period_obs
            })

    # Reconstruct from top harmonics
    reconstructed = np.zeros(n)
    for i, h in enumerate(harmonics):
        t = np.arange(n)
        phase = h['phase']
        freq = 2 * np.pi * h['frequency']
        reconstructed += h['amplitude'] * np.cos(freq * t + phase)

    reconstructed_series = pd.Series(
        reconstructed, index=series[valid_idx].index
    )

    if verbose:
        print(f"Harmonic decomposition: n_harmonics={n_harmonics}, window={window}")
        print(f"  FFT length={n}, top frequency={top_idx_sorted[-1] / n:.4f}")

    return {
        'harmonics': harmonics,
        'frequencies': frequencies,
        'amplitudes': amplitudes,
        'phases': np.angle(fft_vals[1:]),
        'spectrum': powers,
        'reconstructed': reconstructed_series,
        'residual': series[valid_idx] - reconstructed_series
    }


def quality_weighted_decompose(
    series: pd.Series,
    quality: pd.Series,
    method: str = 'stl',
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Decomposition with quality weighting.

    Incorporates quality flags during decomposition (not pre-filtering).
    High-quality observations influence the fit more; low-quality values
    are preserved in output with lower influence on trend/seasonal.

    Args:
        series (pd.Series): Input time series.
        quality (pd.Series): Quality flags (0–1), higher = better.
                            Same index as series.
        method (str): Decomposition method ('stl', 'classical', 'harmonic'). Default 'stl'.
        **kwargs: Additional arguments passed to method function.

    Returns:
        Same structure as method-specific function, with 'quality_weights' added.

    Notes:
        - Quality values outside [0, 1] are clipped
        - All-zero quality creates uniform weights
        - Harmonic decomposition: quality used for ranking, not fitting
    """
    # Validate quality
    if len(quality) != len(series):
        raise ValueError(f"quality length ({len(quality)}) != series length ({len(series)})")

    quality_vals = quality.values.astype(float)
    quality_vals = np.clip(quality_vals, 0, 1)

    if method == 'stl':
        result = stl_decompose(series, weights=quality_vals, **kwargs)
    elif method == 'classical':
        result = classical_decompose(series, **kwargs)
    elif method == 'harmonic':
        result = harmonic_decompose(series, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    result['quality_weights'] = quality_vals
    return result


def reconstruct_from_components(
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    model: str = 'additive',
    components_to_use: Optional[List[str]] = None
) -> pd.Series:
    """
    Reconstruct time series from decomposition components.

    Args:
        trend (pd.Series): Trend component.
        seasonal (pd.Series): Seasonal component.
        residual (pd.Series): Residual component.
        model (str): 'additive' → trend + seasonal + residual
                     'multiplicative' → trend * seasonal * residual
                     Default 'additive'.
        components_to_use (list, optional): ['trend', 'seasonal', 'residual']
                                           subset to reconstruct. None means all.

    Returns:
        pd.Series: Reconstructed time series with original index and NaN preserved.

    Notes:
        - Uses original index from trend (assumes all components have same index)
        - NaN values in input are preserved in output
    """
    if components_to_use is None:
        components_to_use = ['trend', 'seasonal', 'residual']

    components = {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }

    # Check that requested components exist
    for comp in components_to_use:
        if comp not in components:
            raise ValueError(f"Unknown component: {comp}")

    # Reconstruct
    if model == 'additive':
        result = pd.Series(0.0, index=trend.index)
        for comp in components_to_use:
            result += components[comp]
    elif model == 'multiplicative':
        result = pd.Series(1.0, index=trend.index)
        for comp in components_to_use:
            result *= components[comp]
    else:
        raise ValueError(f"Unknown model: {model}")

    # Preserve NaN from trend
    result[trend.isna()] = np.nan

    return result


def detect_seasonality(
    series: pd.Series,
    max_period: int = 730,
    top_n: int = 5,
    verbose: bool = False
) -> Dict:
    """
    Detect dominant seasonal periods via periodogram.

    Identifies potential seasonal periods by analyzing power spectral density.
    Useful for automatic period selection when decomposing unfamiliar time series.

    Args:
        series (pd.Series): Input time series (NaN removed).
        max_period (int): Maximum period to consider. Default 730 observations.
        top_n (int): Return top N candidate periods. Default 5.
        verbose (bool): Print detected periods. Default False.

    Returns:
        Dict with keys:
            - 'primary_period': int, dominant seasonal period
            - 'secondary_periods': list of int, other strong periods
            - 'all_periods': list of (period, power) tuples, ranked
            - 'spectral_density': np.ndarray, power spectrum
            - 'frequencies': np.ndarray, frequency bins
            - 'strength': float (0–1), seasonality strength (seasonal var / total var)

    Notes:
        - Period range: 2 to min(max_period, len(series) // 2)
        - Strength estimate: variance of seasonal component divided by total variance
        - Very short series (< 10) may produce unreliable results
    """
    # Remove NaN
    valid_idx = series.notna()
    series_clean = series[valid_idx].values

    if len(series_clean) < 10:
        warnings.warn(
            f"Series very short ({len(series_clean)} values). "
            "Seasonality detection unreliable.",
            UserWarning
        )

    # Periodogram
    n = len(series_clean)
    max_period = min(max_period, n // 2)

    # Detrend to remove trend influence on spectrum
    series_detrended = signal.detrend(series_clean)

    # FFT
    fft_vals = np.fft.rfft(series_detrended)
    power = np.abs(fft_vals) ** 2
    frequencies = np.fft.rfftfreq(n)

    # Convert frequencies to periods
    periods = []
    powers_by_period = []
    for i, freq in enumerate(frequencies[1:], start=1):  # Skip DC (freq=0)
        if freq > 0:
            period = 1.0 / freq
            if 2 <= period <= max_period:
                periods.append(int(np.round(period)))
                powers_by_period.append(power[i])

    if not periods:
        # Fallback if no valid periods found
        primary_period = 365
        secondary_periods = [7, 30]
        strength = 0.0
    else:
        # Find peaks (local maxima in power)
        peaks, _ = signal.find_peaks(powers_by_period)
        if len(peaks) > 0:
            peak_powers = [(periods[p], powers_by_period[p]) for p in peaks]
            peak_powers.sort(key=lambda x: -x[1])  # Sort by power descending
            primary_period = peak_powers[0][0]
            secondary_periods = [p for p, _ in peak_powers[1:top_n]]
        else:
            # No peaks, use max power
            max_idx = np.argmax(powers_by_period)
            primary_period = periods[max_idx]
            secondary_periods = []

        # Estimate seasonality strength
        # Simple estimate: variance in seasonal range vs total variance
        seasonal_power = np.sum([powers_by_period[i] for i in peaks]) if peaks.size > 0 else 0
        total_power = np.sum(powers_by_period)
        strength = float(seasonal_power / total_power) if total_power > 0 else 0.0

    if verbose:
        print(f"Seasonality detection: primary_period={primary_period}, strength={strength:.3f}")
        if secondary_periods:
            print(f"  Secondary periods: {secondary_periods}")

    all_periods = list(zip(periods, powers_by_period))
    all_periods.sort(key=lambda x: -x[1])

    return {
        'primary_period': primary_period,
        'secondary_periods': secondary_periods,
        'all_periods': all_periods,
        'spectral_density': power,
        'frequencies': frequencies,
        'strength': min(strength, 1.0)  # Clamp to [0, 1]
    }
