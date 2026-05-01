"""
Seasonal-Trend Decomposition for Time Series Analysis

Separates time series into:
- Trend: Long-term direction (slow changes)
- Seasonal: Recurring patterns (daily, weekly, annual cycles)
- Residual: Noise and anomalies

Features:
- Multiple decomposition methods (STL, classical, harmonic)
- Quality-weighted fitting (incorporates data quality flags)
- Lazy evaluation (components computed on first access)
- Handles data gaps and non-stationary behavior

Typical usage:
    decomp = SeasonalTrendDecomposition(
        nee_series, quality=quality_flags, method='stl'
    )
    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.residual
"""

from typing import Optional, List, Dict

import pandas as pd

from diive.core.times.decomposition_utils import (
    stl_decompose, classical_decompose, harmonic_decompose,
    quality_weighted_decompose, reconstruct_from_components,
    detect_seasonality
)


class SeasonalTrendDecomposition:
    """
    Decompose time series into seasonal, trend, and residual components.

    Supports multiple decomposition methods optimized for different use cases:
    - STL (default): Robust, handles gaps and non-stationary data
    - Classical: Simple moving-average based, assumes stationarity
    - Harmonic: Fourier-based, reveals frequency-domain structure

    Lazy evaluation: Components are computed once on first access and cached.

    Example:
        See `examples/analyses/seasonaltrend.py` for complete examples.
    """

    def __init__(
            self,
            series: pd.Series,
            quality: Optional[pd.Series] = None,
            timestamp: Optional[pd.DatetimeIndex] = None,
            method: str = 'stl',
            seasonal_period: Optional[int] = None,
            trend_window: Optional[int] = None,
            robust: bool = True,
            quality_weighted: bool = True,
            seasonal_deg: int = 1,
            trend_deg: int = 1,
            seasonal_jump: int = 1,
            trend_jump: int = 1,
            n_harmonics: int = 10,
            harmonic_window: str = 'hamming',
            verbose: bool = False
    ):
        """
        Initialize SeasonalTrendDecomposition.

        Args:
            series (pd.Series): Time series to decompose (may contain NaN).
            quality (pd.Series, optional): Quality flags (0–1) for each value.
                                          None = all equal weight (1.0).
                                          Same index as series.
            timestamp (pd.DatetimeIndex, optional): DateTime index. Inferred from series.index if None.
            method (str): Decomposition method:
                         'stl' (default) — Seasonal-Trend Loess (robust, gap-tolerant)
                         'classical' — Moving-average (simple, stationary-only)
                         'harmonic' — Fourier-based (frequency analysis)
            seasonal_period (int, optional): Seasonal period in observations.
                                            Auto-detected if None via periodogram.
                                            Example: 365 for daily data (annual cycle)
            trend_window (int, optional): Trend smoothing window length.
                                         Default: 2 * seasonal_period
            robust (bool): Use robust fitting (less sensitive to outliers). Default True.
                          Only applies to STL method.
            quality_weighted (bool): Incorporate quality flags during decomposition.
                                    Default True.
            seasonal_deg (int): Loess polynomial degree for seasonal (0 or 1). Default 1.
            trend_deg (int): Loess polynomial degree for trend (0 or 1). Default 1.
            seasonal_jump (int): STL seasonal jump size (speed optimization). Default 1.
            trend_jump (int): STL trend jump size (speed optimization). Default 1.
            n_harmonics (int): Number of harmonics for harmonic method. Default 10.
            harmonic_window (str): Window function for harmonic ('hamming', 'hann', 'blackman').
                                  Default 'hamming'.
            verbose (bool): Print decomposition details. Default False.

        Raises:
            ValueError: If series is empty or method is invalid.
            TypeError: If series not pd.Series.
        """
        # Input validation
        if not isinstance(series, pd.Series):
            raise TypeError(f"series must be pd.Series, got {type(series)}")

        if len(series) == 0:
            raise ValueError("series cannot be empty")

        if method not in ['stl', 'classical', 'harmonic']:
            raise ValueError(f"method must be 'stl', 'classical', or 'harmonic', got '{method}'")

        self.series = series
        self.quality = quality
        self.timestamp = timestamp if timestamp is not None else series.index
        self.method = method
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window
        self.robust = robust
        self.quality_weighted = quality_weighted and (quality is not None)
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.n_harmonics = n_harmonics
        self.harmonic_window = harmonic_window
        self.verbose = verbose

        # Cached decomposition result
        self._decomposition = None
        self._seasonality_strength = None
        self._seasonal_period_detected = None
        self._detection_result = None

        if self.verbose:
            print(f"SeasonalTrendDecomposition initialized: method={method}, "
                  f"quality_weighted={self.quality_weighted}, verbose={verbose}")

    @property
    def seasonal(self) -> pd.Series:
        """
        Seasonal component (recurring patterns).

        Returns:
            pd.Series with seasonal component. Same index as input series.
                      NaN where input was NaN.
        """
        if self._decomposition is None:
            self._compute_decomposition()
        return self._decomposition['seasonal']

    @property
    def trend(self) -> pd.Series:
        """
        Trend component (long-term direction).

        Returns:
            pd.Series with trend component. Same index as input series.
                      NaN where input was NaN (or by design for classical method).
        """
        if self._decomposition is None:
            self._compute_decomposition()
        return self._decomposition['trend']

    @property
    def residual(self) -> pd.Series:
        """
        Residual component (noise and anomalies).

        Returns:
            pd.Series with residual component. Same index as input series.
                      NaN where input was NaN.
        """
        if self._decomposition is None:
            self._compute_decomposition()
        return self._decomposition['residual']

    @property
    def seasonality_strength(self) -> float:
        """
        Strength of seasonality (0–1).

        Ratio of seasonal variance to total variance of trend + residual.
        Higher = stronger recurring patterns.

        Returns:
            float: Seasonality strength, 0.0–1.0
        """
        if self._seasonality_strength is None:
            seasonal = self.seasonal
            residual = self.residual
            var_residual = residual.var()
            var_seasonal = seasonal.var()
            total = var_residual + var_seasonal
            self._seasonality_strength = (
                var_seasonal / total if total > 0 else 0.0
            )
        return self._seasonality_strength

    def detrend(self) -> pd.Series:
        """
        Remove trend component from series.

        Returns:
            pd.Series: seasonal + residual (trend removed)
        """
        return self.seasonal + self.residual

    def deseasonalize(self) -> pd.Series:
        """
        Remove seasonal component from series.

        Returns:
            pd.Series: trend + residual (seasonality removed)
        """
        return self.trend + self.residual

    def reconstruct(
            self,
            keep_components: Optional[List[str]] = None,
            model: str = 'additive'
    ) -> pd.Series:
        """
        Reconstruct series from selected components.

        Args:
            keep_components (list, optional): Components to include.
                                             Options: 'trend', 'seasonal', 'residual'
                                             Default None = all components
            model (str): Combination model:
                        'additive' (default) → sum components
                        'multiplicative' → multiply components

        Returns:
            pd.Series: Reconstructed series
        """
        if keep_components is None:
            keep_components = ['trend', 'seasonal', 'residual']

        return reconstruct_from_components(
            self.trend, self.seasonal, self.residual,
            model=model, components_to_use=keep_components
        )

    def summary(self) -> str:
        """
        Text summary of decomposition results.

        Returns:
            str: Formatted summary with method, period, strength, and component statistics
        """
        period = self.seasonal_period or self._get_seasonal_period()

        lines = [
            "=" * 60,
            "Seasonal-Trend Decomposition Summary",
            "=" * 60,
            f"Method: {self.method.upper()}",
            f"Seasonal period: {period} observations",
            f"Quality-weighted: {self.quality_weighted}",
            f"Series length: {len(self.series)} ({self.series.notna().sum()} valid)",
            "",
            f"Seasonality strength: {self.seasonality_strength:.3f}",
            "",
            "Component statistics:",
            "-" * 40,
            f"  Trend (mean ± std):      {self.trend.mean():8.3f} ± {self.trend.std():8.3f}",
            f"  Seasonal (mean ± std):   {self.seasonal.mean():8.3f} ± {self.seasonal.std():8.3f}",
            f"  Residual (mean ± std):   {self.residual.mean():8.3f} ± {self.residual.std():8.3f}",
            "",
            f"Original series (mean ± std): {self.series.mean():8.3f} ± {self.series.std():8.3f}",
            "=" * 60
        ]

        return "\n".join(lines)

    def _compute_decomposition(self):
        """
        Compute decomposition using selected method.

        Called automatically on first access to trend, seasonal, or residual properties.
        Result is cached for efficiency.
        """
        if self.verbose:
            print(f"Computing {self.method} decomposition...")

        # Get seasonal period
        period = self._get_seasonal_period()

        # Prepare keyword arguments for the decomposition method
        kwargs = {}

        if self.method == 'stl':
            kwargs = {
                'seasonal': period,
                'trend': self.trend_window or (2 * period + 1),
                'robust': self.robust,
                'seasonal_deg': self.seasonal_deg,
                'trend_deg': self.trend_deg,
                'seasonal_jump': self.seasonal_jump,
                'trend_jump': self.trend_jump,
                'verbose': self.verbose
            }

            if self.quality_weighted and self.quality is not None:
                self._decomposition = quality_weighted_decompose(
                    self.series, self.quality, method='stl', **kwargs
                )
            else:
                self._decomposition = stl_decompose(self.series, **kwargs)

        elif self.method == 'classical':
            kwargs = {
                'period': period,
                'verbose': self.verbose
            }

            self._decomposition = classical_decompose(self.series, **kwargs)

        elif self.method == 'harmonic':
            kwargs = {
                'n_harmonics': self.n_harmonics,
                'period': period,
                'window': self.harmonic_window,
                'verbose': self.verbose
            }

            harmonic_result = harmonic_decompose(self.series, **kwargs)
            # Convert harmonic result to standard format (seasonal=reconstructed, trend=0, residual=residual)
            self._decomposition = {
                'seasonal': harmonic_result['reconstructed'],
                'trend': pd.Series(0.0, index=self.series.index),
                'residual': harmonic_result['residual']
            }

        if self.verbose:
            print(f"Decomposition complete. Seasonality strength: {self.seasonality_strength:.3f}")

    def _get_seasonal_period(self) -> int:
        """
        Get seasonal period, auto-detecting if not provided.

        Returns:
            int: Seasonal period in observations
        """
        if self.seasonal_period is not None:
            return self.seasonal_period

        # Auto-detect
        if self._detection_result is None:
            self._detection_result = detect_seasonality(
                self.series,
                max_period=len(self.series) // 2,
                verbose=self.verbose
            )
            self._seasonal_period_detected = self._detection_result['primary_period']

        return self._seasonal_period_detected

    def detection_results(self) -> Dict:
        """
        Get seasonality detection results (if period was auto-detected).

        Returns:
            Dict with keys:
                - 'primary_period': int, dominant seasonal period
                - 'secondary_periods': list of int, other strong periods
                - 'strength': float (0–1), seasonality strength
                - 'all_periods': list of (period, power) tuples, ranked
        """
        if self._detection_result is None:
            self._detection_result = detect_seasonality(
                self.series,
                max_period=len(self.series) // 2,
                verbose=False
            )

        return {
            'primary_period': self._detection_result['primary_period'],
            'secondary_periods': self._detection_result['secondary_periods'],
            'strength': self._detection_result['strength'],
            'all_periods': self._detection_result['all_periods']
        }

    def __repr__(self) -> str:
        """String representation."""
        period = self.seasonal_period or 'auto'
        return (
            f"SeasonalTrendDecomposition("
            f"method='{self.method}', period={period}, "
            f"quality_weighted={self.quality_weighted})"
        )

    def __str__(self) -> str:
        """String representation (alias for summary)."""
        return self.summary()
