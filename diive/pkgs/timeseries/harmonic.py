"""
Harmonic (Fourier-based) time series analysis utilities.

Functions for decomposing time series into sine/cosine basis functions,
extracting amplitude and phase information, and analyzing frequency-domain
properties.

Uses FFT (Fast Fourier Transform) for efficient computation.
"""

import numpy as np
import pandas as pd
from scipy import signal, fft as scipy_fft
from typing import Dict, Tuple, Optional, List


def harmonic_analysis(
    series: pd.Series,
    period: int,
    n_harmonics: int = 10,
    window: str = 'hamming',
    verbose: bool = False
) -> Dict:
    """
    Extract amplitude and phase of harmonic components.

    Performs FFT and identifies harmonics at multiples of the fundamental
    frequency (1/period).

    Args:
        series (pd.Series): Input time series (NaN values removed).
        period (int): Fundamental period in observations.
        n_harmonics (int): Number of harmonics to extract. Default 10.
        window (str): Window function ('hamming', 'hann', 'blackman'). Default 'hamming'.
        verbose (bool): Print analysis details. Default False.

    Returns:
        Dict with keys:
            - 'fundamental_frequency': float, fundamental frequency (1/period)
            - 'harmonics': list of dicts with 'amplitude', 'phase', 'frequency', 'harmonic_number'
            - 'frequencies': np.ndarray, all frequency bins
            - 'amplitudes': np.ndarray, amplitude at each frequency
            - 'phases': np.ndarray, phase at each frequency (radians)
            - 'power': np.ndarray, power (amplitude squared)

    Notes:
        - Window function reduces spectral leakage
        - Phase is in radians [-π, π]
        - Harmonics are ordered by harmonic number (1st, 2nd, 3rd, etc.)
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
    amplitudes = 2 * np.abs(fft_vals[1:])  # Double for one-sided spectrum, exclude DC
    phases = np.angle(fft_vals[1:])
    power = amplitudes ** 2

    # Fundamental frequency
    fundamental_freq = 1.0 / period if period > 0 else 0.0

    # Extract harmonics at multiples of fundamental frequency
    harmonics = []
    for h_num in range(1, n_harmonics + 1):
        target_freq = h_num * fundamental_freq
        # Find closest FFT bin
        if target_freq < frequencies[-1]:
            idx = int(np.round(target_freq * n))
            if idx < len(amplitudes):
                harmonics.append({
                    'harmonic_number': h_num,
                    'target_frequency': target_freq,
                    'actual_frequency': frequencies[idx + 1],  # +1 for DC exclusion
                    'amplitude': amplitudes[idx],
                    'phase': phases[idx],
                    'power': power[idx]
                })

    if verbose:
        print(f"Harmonic analysis: period={period}, fundamental_freq={fundamental_freq:.6f}")
        print(f"  Extracted {len(harmonics)} harmonics, window={window}")

    return {
        'fundamental_frequency': fundamental_freq,
        'harmonics': harmonics,
        'frequencies': frequencies,
        'amplitudes': np.concatenate([[0], amplitudes]),  # Include DC bin
        'phases': np.concatenate([[0], phases]),
        'power': np.concatenate([[0], power])
    }


def reconstruct_harmonics(
    harmonics: List[Dict],
    n_samples: int
) -> np.ndarray:
    """
    Reconstruct signal from harmonic components.

    Args:
        harmonics (list): List of harmonic dicts with 'amplitude', 'phase', 'frequency'.
        n_samples (int): Length of reconstructed signal.

    Returns:
        np.ndarray: Reconstructed signal as 1D array.

    Notes:
        - Each harmonic contributes: amplitude * cos(2π * frequency * t + phase)
        - Components summed additively
    """
    reconstructed = np.zeros(n_samples)
    t = np.arange(n_samples)

    for h in harmonics:
        amplitude = h.get('amplitude', 0)
        phase = h.get('phase', 0)
        frequency = h.get('frequency', h.get('actual_frequency', 0))

        reconstructed += amplitude * np.cos(2 * np.pi * frequency * t + phase)

    return reconstructed


def periodogram(
    series: pd.Series,
    detrend_method: str = 'linear',
    return_power: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Compute power spectral density via periodogram.

    Uses Welch's method for robust power estimation (reduces variance compared to raw FFT).

    Args:
        series (pd.Series): Input time series (NaN removed).
        detrend_method (str): Detrend method ('linear', 'constant', None). Default 'linear'.
        return_power (bool): Return power (amplitude^2) or amplitude. Default True.
        verbose (bool): Print periodogram details. Default False.

    Returns:
        Dict with keys:
            - 'frequencies': np.ndarray, frequency bins
            - 'power' or 'amplitude': np.ndarray, spectral density
            - 'peaks': list of (frequency, power/amplitude) for local maxima
            - 'dominant_frequency': float, frequency with maximum power

    Notes:
        - Welch method uses overlapping windows; more robust than raw FFT
        - Frequencies range from 0 to 0.5 (Nyquist frequency)
        - Peaks detected as local maxima in power
    """
    # Remove NaN
    valid_idx = series.notna()
    series_clean = series[valid_idx].values

    if len(series_clean) < 4:
        raise ValueError(f"Series must have >= 4 valid values, got {len(series_clean)}")

    # Detrend
    if detrend_method == 'linear':
        series_clean = signal.detrend(series_clean, type='linear')
    elif detrend_method == 'constant':
        series_clean = signal.detrend(series_clean, type='constant')
    # else: no detrending

    # Welch periodogram
    frequencies, power = signal.welch(
        series_clean,
        nperseg=min(len(series_clean), 256),
        scaling='spectrum'
    )

    # Find peaks
    peaks, peak_props = signal.find_peaks(power, prominence=0.01 * power.max() if power.max() > 0 else 0.01)
    peak_list = [(frequencies[p], power[p]) for p in peaks]
    peak_list.sort(key=lambda x: -x[1])  # Sort by power descending

    # Dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = frequencies[dominant_idx]

    if verbose:
        print(f"Periodogram: {len(frequencies)} frequency bins, detrend={detrend_method}")
        print(f"  Dominant frequency: {dominant_freq:.6f} (period ~{1/dominant_freq:.1f} if > 0)")
        print(f"  Found {len(peaks)} spectral peaks")

    power_key = 'power' if return_power else 'amplitude'

    return {
        'frequencies': frequencies,
        power_key: power,
        'peaks': peak_list,
        'dominant_frequency': dominant_freq,
        'dominant_period': 1.0 / dominant_freq if dominant_freq > 0 else np.inf
    }


def fft_decompose(
    series: pd.Series,
    n_components: int = 10,
    window: str = 'hamming'
) -> Dict:
    """
    Decompose signal into top Fourier components by power.

    Args:
        series (pd.Series): Input time series (NaN removed).
        n_components (int): Number of components to extract. Default 10.
        window (str): Window function. Default 'hamming'.

    Returns:
        Dict with keys:
            - 'components': list of (frequency, amplitude, phase, power)
            - 'reconstructed': pd.Series, signal reconstructed from components
            - 'residual': pd.Series, error (original - reconstructed)
            - 'explained_variance_ratio': list, cumulative variance explained by each component
            - 'total_variance': float, original signal variance

    Notes:
        - Components ranked by power (amplitude squared)
        - Explained variance ratio: cumsum of component power / total power
    """
    # Remove NaN
    valid_idx = series.notna()
    series_clean = series[valid_idx].values

    if len(series_clean) < 4:
        raise ValueError(f"Series must have >= 4 valid values, got {len(series_clean)}")

    n = len(series_clean)

    # Apply window
    try:
        window_func = signal.get_window(window, n)
    except ValueError:
        window_func = signal.hamming(n)

    series_windowed = series_clean * window_func

    # FFT
    frequencies = np.fft.rfftfreq(n)
    fft_vals = np.fft.rfft(series_windowed) / n
    amplitudes = 2 * np.abs(fft_vals[1:])
    phases = np.angle(fft_vals[1:])
    power = amplitudes ** 2

    # Top components by power
    top_idx = np.argsort(-power)[:min(n_components, len(power))]

    components = []
    for idx in top_idx:
        components.append({
            'frequency': frequencies[idx + 1],  # +1 for DC
            'period': 1.0 / frequencies[idx + 1] if frequencies[idx + 1] > 0 else np.inf,
            'amplitude': amplitudes[idx],
            'phase': phases[idx],
            'power': power[idx]
        })

    # Sort by frequency for reconstruction
    components_sorted = sorted(components, key=lambda x: x['frequency'])

    # Reconstruct
    reconstructed = np.zeros(n)
    for c in components_sorted:
        t = np.arange(n)
        reconstructed += c['amplitude'] * np.cos(2 * np.pi * c['frequency'] * t + c['phase'])

    # Explained variance
    total_power = np.sum(power)
    component_powers = [c['power'] for c in components]
    cumsum_power = np.cumsum(component_powers)
    explained_var = cumsum_power / total_power if total_power > 0 else cumsum_power

    reconstructed_series = pd.Series(reconstructed, index=series[valid_idx].index)
    residual = series[valid_idx] - reconstructed_series

    return {
        'components': components_sorted,
        'reconstructed': reconstructed_series,
        'residual': residual,
        'explained_variance_ratio': explained_var.tolist(),
        'total_variance': float(np.var(series_clean))
    }


def multi_scale_harmonics(
    series: pd.Series,
    periods: List[int],
    n_harmonics_per_period: int = 3,
    window: str = 'hamming'
) -> Dict:
    """
    Analyze harmonics at multiple scales (e.g., diurnal, annual, etc.).

    Useful for separating seasonal components at different time scales
    (daily, weekly, annual patterns).

    Args:
        series (pd.Series): Input time series.
        periods (list of int): Periods to analyze (e.g., [24, 7, 365] for hours/days/years).
        n_harmonics_per_period (int): Harmonics per period. Default 3.
        window (str): Window function. Default 'hamming'.

    Returns:
        Dict with keys:
            - 'scales': list of dicts, each containing:
                - 'period': int
                - 'harmonics': list of harmonic dicts
                - 'total_power': float
                - 'strength': float (0–1, power ratio)

    Notes:
        - Strength at each scale = sum of harmonic powers / total signal power
        - Useful for understanding multi-scale seasonality (e.g., daily + annual)
    """
    # Total signal power (for strength calculation)
    valid_idx = series.notna()
    series_clean = series[valid_idx].values
    total_signal_power = np.var(series_clean)

    scales = []
    for period in periods:
        try:
            h_analysis = harmonic_analysis(
                series, period=period, n_harmonics=n_harmonics_per_period, window=window
            )

            total_period_power = sum(h['power'] for h in h_analysis['harmonics'])
            strength = total_period_power / total_signal_power if total_signal_power > 0 else 0.0

            scales.append({
                'period': period,
                'harmonics': h_analysis['harmonics'],
                'total_power': total_period_power,
                'strength': min(strength, 1.0)
            })
        except Exception as e:
            # Skip periods that fail
            continue

    return {'scales': scales}
