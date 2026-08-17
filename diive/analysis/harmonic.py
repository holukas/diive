"""
ANALYSIS: HARMONIC DECOMPOSITION
=================================

Fourier-based time series decomposition using FFT for frequency-domain analysis.
Extracts amplitude and phase information from sine/cosine basis functions.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict

from diive.core.utils.console import detail


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
        - Amplitudes are divided by the window's coherent gain (its mean), so the
          taper does not scale them down and a given signal returns the same
          amplitude under any window
        - Phase is in radians [-π, π]
        - Harmonics are ordered by harmonic number (1st, 2nd, 3rd, etc.)

    Example:
        See `examples/analysis/analysis_harmonic.py` for diel and annual harmonic
        decomposition of flux and meteo data, plus the effect of the window.
    """
    # Remove NaN
    valid_idx = series.notna()
    series_clean = series[valid_idx].values

    if len(series_clean) < 4:
        raise ValueError(f"Series must have >= 4 valid values, got {len(series_clean)}")

    # Apply window
    window_func = signal.get_window(window, len(series_clean))

    series_windowed = series_clean * window_func

    # FFT
    n = len(series_windowed)
    frequencies = np.fft.rfftfreq(n)
    fft_vals = np.fft.rfft(series_windowed) / n
    # Divide by the window's coherent gain (its mean). A window scales the signal
    # down - the default hamming has a mean of ~0.54 - so without this every
    # reported amplitude is that fraction of the true one, and a reconstruction
    # built from them is short by the rest.
    coherent_gain = float(np.mean(window_func))
    amplitudes = 2 * np.abs(fft_vals[1:]) / coherent_gain  # one-sided, DC excluded
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
            # `bin_full` indexes the full rfft output; amplitudes/phases/power
            # have DC stripped, so they are one shorter and need bin_full - 1.
            # Reading them at `bin_full` returned the neighbouring bin: a pure
            # cosine sitting exactly on its bin came back with amplitude 0.
            bin_full = int(np.round(target_freq * n))
            idx = bin_full - 1
            if 0 <= idx < len(amplitudes):
                harmonics.append({
                    'harmonic_number': h_num,
                    'target_frequency': target_freq,
                    'actual_frequency': frequencies[bin_full],
                    'amplitude': amplitudes[idx],
                    'phase': phases[idx],
                    'power': power[idx]
                })

    if verbose:
        detail(f"Harmonic analysis: period={period}, fundamental_freq={fundamental_freq:.6f}, "
               f"extracted {len(harmonics)} harmonics, window={window}", verbose=verbose)

    return {
        'fundamental_frequency': fundamental_freq,
        'harmonics': harmonics,
        'frequencies': frequencies,
        'amplitudes': np.concatenate([[0], amplitudes]),  # Include DC bin
        'phases': np.concatenate([[0], phases]),
        'power': np.concatenate([[0], power])
    }


def spectrogram(
    series: pd.Series,
    nperseg: int = 256,
    noverlap: int = None,
    window: str = 'hann',
    scaling: str = 'spectrum',
    detrend: str = 'constant',
    verbose: bool = False,
) -> Dict:
    """
    Spectrogram (short-time Fourier transform): how the frequency content of a
    series evolves over time.

    Splits the series into overlapping windows and computes a spectrum for each,
    revealing *when* each cyclic component is strong. For eddy-covariance data
    this shows, e.g., the 1-cycle-per-day photosynthesis rhythm strengthening in
    the growing season and fading in winter — information a single static
    spectrum (:func:`harmonic_analysis`) cannot show.

    Args:
        series (pd.Series): Input time series (NaN values removed).
        nperseg (int): Samples per segment (window length). Larger = finer
            frequency resolution but coarser time resolution. Clamped to the
            series length. Default 256.
        noverlap (int): Samples of overlap between segments. Default nperseg // 2.
        window (str): Window function ('hann', 'hamming', 'blackman', ...). Default 'hann'.
        scaling (str): 'spectrum' (power spectrum) or 'density' (power spectral
            density). Default 'spectrum'.
        detrend (str | bool): Per-segment detrending ('constant', 'linear', or
            False). Default 'constant'.
        verbose (bool): Print details. Default False.

    Returns:
        Dict with keys:
            - 'frequencies': np.ndarray, frequency bins (cycles per record)
            - 'times': np.ndarray, segment-centre positions (records from start)
            - 'power': np.ndarray, 2D [n_frequencies, n_times] power
            - 'power_db': np.ndarray, 10*log10(power) for plotting (decibels)

    Notes:
        - Frequencies are in cycles per record; multiply by the sampling rate to
          interpret (e.g. x48 for cycles/day on half-hourly data).
        - ``nperseg`` trades frequency resolution against time resolution.
    """
    valid = series.notna()
    values = series[valid].to_numpy()

    if len(values) < 4:
        raise ValueError(f"Series must have >= 4 valid values, got {len(values)}")

    nperseg = min(nperseg, len(values))
    if noverlap is None:
        noverlap = nperseg // 2

    frequencies, times, power = signal.spectrogram(
        values, nperseg=nperseg, noverlap=noverlap, window=window,
        scaling=scaling, detrend=detrend,
    )
    power_db = 10.0 * np.log10(power + 1e-12)

    if verbose:
        detail(f"Spectrogram: nperseg={nperseg}, noverlap={noverlap}, window={window}, "
               f"{power.shape[0]} freqs x {power.shape[1]} time segments", verbose=verbose)

    return {
        'frequencies': frequencies,
        'times': times,
        'power': power,
        'power_db': power_db,
    }


def spectrogram_to_code(
        varname: str,
        *,
        nperseg: int = 256,
        noverlap: int | None = None,
        window: str = 'hann',
        max_cycles_per_day: float | None = None,
        cmap: str = 'viridis',
        df_name: str = 'df',
) -> str:
    """Render a runnable :func:`spectrogram` snippet (compute + time-frequency plot).

    Mirrors what the GUI's Spectrogram tab shows: the short-time Fourier transform
    over the series, mapped onto calendar-time x cycles-per-day axes and drawn with
    ``pcolormesh`` (the segment centres mapped back to real timestamps so the
    x-axis stays calendar time even across gaps). Belongs in the library (not the
    GUI): it encodes the exact call shape and must stay correct as that API
    evolves; the GUI only calls it (the GUI <-> library separation rule).

    Args:
        varname: Column to analyse.
        nperseg / noverlap / window: passed straight to :func:`spectrogram`.
        max_cycles_per_day: upper limit of the frequency (y) axis, or None.
        cmap: matplotlib colormap name.
        df_name: variable name used for the input DataFrame.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = [
        "import matplotlib.dates as mdates",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "import diive as dv",
        "",
        f"series = {df_name}[{varname!r}]",
        "spec = dv.analysis.spectrogram(",
        "    series,",
        f"    nperseg={nperseg!r},",
        f"    noverlap={noverlap!r},",
        f"    window={window!r},",
        ")",
        "",
        "# Map each segment centre (in valid-sample positions) to its real",
        "# timestamp, so the x-axis is calendar time even across gaps.",
        "valid_index = series.dropna().index",
        f"delta = pd.Series({df_name}.index).diff().median()",
        "rec_per_day = pd.Timedelta('1D') / delta",
        "cycles_per_day = spec['frequencies'] * rec_per_day",
        "pos = np.clip(np.round(spec['times']).astype(int), 0, len(valid_index) - 1)",
        "x = mdates.date2num(valid_index[pos].to_pydatetime())",
        "",
        "fig, ax = plt.subplots(figsize=(12, 5))",
        "mesh = ax.pcolormesh(x, cycles_per_day, spec['power_db'],",
        f"                     shading='gouraud', cmap={cmap!r})",
    ]
    if max_cycles_per_day is not None:
        lines.append(f"ax.set_ylim(0, {max_cycles_per_day!r})")
    lines += [
        "ax.axhline(1.0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)",
        "ax.xaxis_date()",
        "ax.xaxis.set_major_formatter(",
        "    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))",
        "ax.set_xlabel('Time')",
        "ax.set_ylabel('Frequency (cycles per day)')",
        f"ax.set_title('Spectrogram - ' + {varname!r})",
        "fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01).set_label('Power (dB)')",
        "plt.show()",
    ]
    return "\n".join(lines) + "\n"
