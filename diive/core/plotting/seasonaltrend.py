"""
Visualization functions for seasonal-trend decomposition.

Provides methods to plot decomposition results, spectral analysis,
and seasonality strength comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple
import warnings

try:
    from statsmodels.graphics.tsaplots import plot_acf
except ImportError:
    plot_acf = None


def plot_decomposition(
    decomp,
    figsize: Tuple[int, int] = (14, 10),
    show_residual_acf: bool = False,
    highlight_quality: bool = False,
    quality: Optional[pd.Series] = None,
    title: Optional[str] = None,
    color_scheme: Optional[dict] = None
) -> plt.Figure:
    """
    Plot decomposition components in 4-panel layout.

    Args:
        decomp: SeasonalTrendDecomposition object
        figsize (tuple): Figure size (width, height). Default (14, 10)
        show_residual_acf (bool): Show ACF of residuals in 4th panel. Default False
        highlight_quality (bool): Color points by quality flag alpha. Default False
        quality (pd.Series, optional): Quality flags (0–1) for transparency.
                                       Used only if highlight_quality=True.
        title (str, optional): Figure title. If None, auto-generated.
        color_scheme (dict, optional): Custom colors {'original', 'trend', 'seasonal', 'residual'}.
                                      If None, uses defaults.

    Returns:
        plt.Figure: Matplotlib figure object

    Notes:
        - Panel 1: Original series (black) with trend overlay (blue)
        - Panel 2: Trend component
        - Panel 3: Seasonal component (shows ~2 cycles)
        - Panel 4: Residuals (or ACF of residuals)
        - Quality highlighting uses alpha transparency (high quality = opaque)
    """
    if color_scheme is None:
        color_scheme = {
            'original': '#1f77b4',  # blue
            'trend': '#d62728',     # red
            'seasonal': '#ff7f0e',  # orange
            'residual': '#7f7f7f',  # gray
            'acf': '#2ca02c'        # green
        }

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, hspace=0.4)

    # Get components
    original = decomp.series
    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.residual

    # Set up alpha values from quality if needed
    alpha_array = None
    if highlight_quality and quality is not None:
        # Scale quality to alpha range [0.3, 1.0]
        quality_vals = quality.fillna(0.5).values
        alpha_array = 0.3 + 0.7 * np.clip(quality_vals, 0, 1)

    # Panel 1: Original with trend overlay
    ax1 = fig.add_subplot(gs[0])
    if highlight_quality and alpha_array is not None:
        for i in range(len(original) - 1):
            ax1.plot(original.index[i:i+2], original.iloc[i:i+2].values,
                    color=color_scheme['original'], alpha=alpha_array[i], linewidth=0.8)
    else:
        ax1.plot(original.index, original.values, color=color_scheme['original'],
                label='Original', linewidth=0.8)

    ax1.plot(trend.index, trend.values, color=color_scheme['trend'],
            label='Trend', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Original + Trend')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or f"Decomposition ({decomp.method.upper()})")

    # Panel 2: Trend
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(trend.index, trend.values, color=color_scheme['trend'], linewidth=1.5)
    ax2.fill_between(trend.index, trend.values, alpha=0.3, color=color_scheme['trend'])
    ax2.set_ylabel('Trend')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Seasonal (show ~2-3 cycles)
    ax3 = fig.add_subplot(gs[2])
    period = decomp.seasonal_period or 365  # Fallback
    end_idx = min(len(seasonal), 3 * period)
    ax3.plot(seasonal.index[:end_idx], seasonal.iloc[:end_idx].values,
            color=color_scheme['seasonal'], linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    ax3.fill_between(seasonal.index[:end_idx], seasonal.iloc[:end_idx].values, alpha=0.3,
                    color=color_scheme['seasonal'])
    ax3.set_ylabel('Seasonal')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residuals or ACF
    ax4 = fig.add_subplot(gs[3])

    if show_residual_acf and plot_acf is not None:
        try:
            residual_clean = residual.dropna()
            plot_acf(residual_clean, lags=min(40, len(residual_clean) // 4),
                    ax=ax4, color=color_scheme['acf'])
            ax4.set_ylabel('ACF')
            ax4.set_title('Residual Autocorrelation')
        except Exception as e:
            warnings.warn(f"Could not plot ACF: {str(e)}")
            # Fallback to residual plot
            ax4.plot(residual.index, residual.values, color=color_scheme['residual'],
                    linewidth=0.8, marker='o', markersize=2)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            ax4.set_ylabel('Residual')
            ax4.grid(True, alpha=0.3)
    else:
        if highlight_quality and alpha_array is not None:
            for i in range(len(residual) - 1):
                ax4.plot(residual.index[i:i+2], residual.iloc[i:i+2].values,
                        color=color_scheme['residual'], alpha=alpha_array[i], linewidth=0.8)
        else:
            ax4.plot(residual.index, residual.values, color=color_scheme['residual'],
                    linewidth=0.8, marker='o', markersize=2)

        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
        ax4.set_ylabel('Residual')
        ax4.grid(True, alpha=0.3)

    ax4.set_xlabel('Time')

    # Format x-axis as dates if datetime index
    for ax in [ax1, ax2, ax3, ax4]:
        if isinstance(original.index, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return fig


def plot_seasonal_strength_by_period(
    series: pd.Series,
    periods: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    color: str = '#1f77b4'
) -> plt.Figure:
    """
    Compare seasonality strength across multiple periods.

    Useful for identifying dominant seasonal cycles at different scales
    (e.g., diurnal, weekly, annual).

    Args:
        series (pd.Series): Input time series
        periods (list, optional): Periods to evaluate. Default [7, 14, 30, 365]
        figsize (tuple): Figure size. Default (10, 6)
        title (str, optional): Figure title
        color (str): Bar color. Default blue

    Returns:
        plt.Figure: Matplotlib figure object

    Notes:
        - Strength is seasonal variance / (seasonal + residual variance)
        - Higher strength = stronger seasonality at that period
        - Dominance across periods suggests multi-scale seasonality
    """
    if periods is None:
        periods = [7, 14, 30, 365]

    # Import here to avoid circular dependency
    from diive.pkgs.analyses.seasonaltrend import SeasonalTrendDecomposition

    # Limit to valid periods
    periods = [p for p in periods if p < len(series)]

    if not periods:
        raise ValueError(f"All periods >= series length ({len(series)})")

    strengths = []
    period_labels = []

    for period in periods:
        try:
            decomp = SeasonalTrendDecomposition(
                series, method='stl', seasonal_period=period, verbose=False
            )
            strengths.append(decomp.seasonality_strength)
            period_labels.append(f"{period}")
        except Exception:
            # Skip periods that fail
            continue

    if not strengths:
        raise ValueError("Could not compute seasonality for any period")

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(strengths)), strengths, color=color, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, strength) in enumerate(zip(bars, strengths)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{strength:.3f}',
               ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(period_labels)))
    ax.set_xticklabels(period_labels)
    ax.set_xlabel('Period (observations)')
    ax.set_ylabel('Seasonality Strength')
    ax.set_ylim(0, min(1.0, max(strengths) * 1.2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(title or 'Seasonality Strength by Period')

    return fig


def plot_harmonics(
    harmonics: List[dict],
    figsize: Tuple[int, int] = (12, 6),
    top_n: int = 5,
    title: Optional[str] = None,
    color: str = '#1f77b4'
) -> plt.Figure:
    """
    Plot Fourier spectrum with labeled harmonics.

    Args:
        harmonics (list): List of harmonic dicts with 'harmonic_number', 'amplitude', 'frequency'
        figsize (tuple): Figure size. Default (12, 6)
        top_n (int): Highlight top N harmonics. Default 5
        title (str, optional): Figure title
        color (str): Bar color. Default blue

    Returns:
        plt.Figure: Matplotlib figure object

    Notes:
        - Harmonics sorted by amplitude (power)
        - Top N labeled with frequency/period information
        - Useful for understanding multi-scale periodicity
    """
    if not harmonics:
        raise ValueError("No harmonics provided")

    # Sort by amplitude descending
    harmonics_sorted = sorted(harmonics, key=lambda h: h['amplitude'], reverse=True)
    top_n = min(top_n, len(harmonics_sorted))

    fig, ax = plt.subplots(figsize=figsize)

    amplitudes = [h['amplitude'] for h in harmonics_sorted[:top_n]]
    harmonic_nums = [h['harmonic_number'] for h in harmonics_sorted[:top_n]]

    bars = ax.bar(range(len(amplitudes)), amplitudes, color=color, alpha=0.7, edgecolor='black')

    # Add labels
    labels = []
    for i, h in enumerate(harmonics_sorted[:top_n]):
        freq = h.get('actual_frequency', h.get('frequency', 0))
        period = 1.0 / freq if freq > 0 else np.inf
        if period < np.inf:
            labels.append(f"H{h['harmonic_number']}\n({period:.1f})")
        else:
            labels.append(f"H{h['harmonic_number']}")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Harmonic')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(title or f'Top {top_n} Fourier Harmonics')

    # Add value labels on bars
    for i, (bar, amp) in enumerate(zip(bars, amplitudes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{amp:.3f}',
               ha='center', va='bottom', fontsize=9)

    return fig


def plot_spectral_density(
    frequencies: np.ndarray,
    power: np.ndarray,
    peaks: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot power spectral density (periodogram).

    Args:
        frequencies (np.ndarray): Frequency bins
        power (np.ndarray): Power at each frequency
        peaks (list, optional): List of (frequency, power) tuples for peaks
        figsize (tuple): Figure size. Default (12, 6)
        title (str, optional): Figure title
        log_scale (bool): Use log scale for power. Default False

    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(frequencies, power, color='#1f77b4', linewidth=1.5, label='PSD')

    if peaks:
        peak_freqs = [f for f, _ in peaks]
        peak_powers = [p for _, p in peaks]
        ax.scatter(peak_freqs, peak_powers, color='red', s=100, marker='x',
                  linewidths=2, label='Peaks')

        # Label top 3 peaks
        for i, (freq, power_val) in enumerate(peaks[:3]):
            period = 1.0 / freq if freq > 0 else np.inf
            ax.annotate(f'T={period:.1f}', xy=(freq, power_val),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or 'Power Spectral Density')
    ax.legend()

    return fig
