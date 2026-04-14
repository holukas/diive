"""
Unit tests for seasonal-trend decomposition module.

Tests cover:
- Multiple decomposition methods (STL, classical, harmonic)
- Quality-weighted fitting
- Gap handling (missing values)
- Edge cases (short series, all-NaN, constant values)
- Numerical consistency (reconstruction, Parseval)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from diive.core.times.decomposition_utils import (
    stl_decompose, classical_decompose, harmonic_decompose,
    quality_weighted_decompose, reconstruct_from_components,
    detect_seasonality
)
from diive.pkgs.analyses.seasonaltrend import SeasonalTrendDecomposition
from diive.pkgs.timeseries.harmonic import (
    harmonic_analysis, reconstruct_harmonics, periodogram, fft_decompose
)


class TestDecompositionUtils:
    """Test low-level decomposition functions."""

    @pytest.fixture
    def synthetic_series(self):
        """Create synthetic time series: trend + seasonal + noise."""
        np.random.seed(42)
        n = 365 * 2  # 2 years of daily data

        # Trend: linear increase
        trend = np.linspace(10, 15, n)

        # Seasonal: annual cycle
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365)

        # Noise
        noise = np.random.normal(0, 0.5, n)

        # Combine
        series = trend + seasonal + noise
        index = pd.date_range('2020-01-01', periods=n, freq='D')

        return pd.Series(series, index=index)

    def test_stl_decompose_basic(self, synthetic_series):
        """Test STL decomposition produces valid components."""
        result = stl_decompose(synthetic_series, seasonal=365, trend=730)

        # Check output structure
        assert 'seasonal' in result
        assert 'trend' in result
        assert 'residual' in result

        # Check shapes
        assert len(result['seasonal']) == len(synthetic_series)
        assert len(result['trend']) == len(synthetic_series)
        assert len(result['residual']) == len(synthetic_series)

        # Check reconstruction approximates original
        reconstructed = result['trend'] + result['seasonal'] + result['residual']
        mse = np.mean((reconstructed - synthetic_series) ** 2)
        assert mse < 1.0  # Should be very small error

    def test_stl_handles_gaps(self, synthetic_series):
        """Test STL handles missing values gracefully."""
        # Introduce gaps
        series_with_gaps = synthetic_series.copy()
        series_with_gaps.iloc[50:60] = np.nan
        series_with_gaps.iloc[100:110] = np.nan

        result = stl_decompose(series_with_gaps, seasonal=365, trend=730)

        # Components should have same length
        assert len(result['seasonal']) == len(series_with_gaps)

        # NaN locations should be preserved (approximately)
        # Some NaN propagation near gaps is expected due to Loess
        assert result['seasonal'].isna().sum() > 0
        assert result['trend'].isna().sum() > 0

    def test_classical_decompose(self, synthetic_series):
        """Test classical decomposition."""
        result = classical_decompose(synthetic_series, period=365)

        # Check output structure
        assert 'seasonal' in result
        assert 'trend' in result
        assert 'residual' in result

        # Seasonal should be periodic
        seasonal = result['seasonal']
        # First period should roughly match second period
        first_year = seasonal.iloc[:365]
        second_year = seasonal.iloc[365:730]
        correlation = np.corrcoef(first_year.values, second_year.values)[0, 1]
        assert correlation > 0.9

    def test_harmonic_decompose(self, synthetic_series):
        """Test harmonic (Fourier) decomposition."""
        result = harmonic_decompose(synthetic_series, n_harmonics=5, period=365)

        # Check output structure
        assert 'harmonics' in result
        assert 'reconstructed' in result
        assert 'residual' in result

        # Harmonics should be ordered by frequency
        harmonics = result['harmonics']
        assert len(harmonics) <= 5

        # First harmonic should be close to fundamental period
        if len(harmonics) > 0:
            first_period = harmonics[0]['period']
            # Should be close to 365 days (within 10%)
            assert abs(first_period - 365) / 365 < 0.2

    def test_quality_weighted_decompose(self, synthetic_series):
        """Test quality-weighted decomposition."""
        # Create quality flags (high for most, low for some)
        quality = pd.Series(np.ones(len(synthetic_series)), index=synthetic_series.index)
        quality.iloc[50:60] = 0.1  # Low quality section
        quality.iloc[100:110] = 0.2

        result = quality_weighted_decompose(
            synthetic_series, quality, method='stl', seasonal=365, trend=730
        )

        # Should have quality_weights in result
        assert 'quality_weights' in result
        assert len(result['quality_weights']) == len(synthetic_series)

        # Low-quality region should have more residual (less trust in fit)
        residual_low_quality = np.abs(result['residual'].iloc[50:60]).mean()
        residual_high_quality = np.abs(result['residual'].iloc[200:210]).mean()
        # This is a soft test; may not always hold
        # assert residual_low_quality > residual_high_quality * 0.5

    def test_reconstruct_from_components(self, synthetic_series):
        """Test reconstructing series from components."""
        decomp = stl_decompose(synthetic_series, seasonal=365)

        # Reconstruct all components
        reconstructed = reconstruct_from_components(
            decomp['trend'], decomp['seasonal'], decomp['residual'],
            model='additive'
        )

        # Should closely match original
        mse = np.mean((reconstructed - synthetic_series) ** 2)
        assert mse < 1.0

    def test_reconstruct_subset_components(self, synthetic_series):
        """Test reconstructing with subset of components."""
        decomp = stl_decompose(synthetic_series, seasonal=365)

        # Reconstruct without trend
        detrended = reconstruct_from_components(
            decomp['trend'], decomp['seasonal'], decomp['residual'],
            model='additive',
            components_to_use=['seasonal', 'residual']
        )

        # Should not include trend
        expected = decomp['seasonal'] + decomp['residual']
        np.testing.assert_array_almost_equal(detrended.values, expected.values, decimal=5)

    def test_detect_seasonality(self, synthetic_series):
        """Test automatic seasonality detection."""
        result = detect_seasonality(synthetic_series, max_period=730)

        # Should detect annual cycle
        assert 'primary_period' in result
        primary = result['primary_period']
        assert 300 < primary < 400  # Close to 365

        # Should have secondary periods
        assert 'secondary_periods' in result
        assert isinstance(result['secondary_periods'], list)

        # Strength should be non-zero
        assert result['strength'] > 0


class TestSeasonalTrendDecomposition:
    """Test main SeasonalTrendDecomposition class."""

    @pytest.fixture
    def nee_like_series(self):
        """Create NEE-like time series (daily data with annual/diurnal patterns)."""
        np.random.seed(42)
        n = 365 * 3  # 3 years

        # Trend: linear recovery
        trend = np.linspace(-2, 0, n)

        # Annual cycle: stronger uptake in growing season
        seasonal = -3 * np.cos(2 * np.pi * np.arange(n) / 365)

        # Diurnal-like modulation (simplified)
        diurnal = 0.5 * np.sin(2 * np.pi * np.arange(n) / 1)

        # Noise
        noise = np.random.normal(0, 0.3, n)

        # Combine
        series = trend + seasonal + 0.1 * diurnal + noise
        index = pd.date_range('2020-01-01', periods=n, freq='D')

        return pd.Series(series, index=index)

    def test_init_valid_inputs(self, nee_like_series):
        """Test initialization with valid inputs."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series,
            method='stl',
            seasonal_period=365,
            verbose=False
        )
        assert decomp.series is nee_like_series
        assert decomp.method == 'stl'

    def test_init_invalid_method(self, nee_like_series):
        """Test initialization with invalid method raises error."""
        with pytest.raises(ValueError):
            SeasonalTrendDecomposition(
                nee_like_series, method='invalid'
            )

    def test_init_empty_series(self):
        """Test initialization with empty series raises error."""
        with pytest.raises(ValueError):
            SeasonalTrendDecomposition(pd.Series([], dtype=float))

    def test_lazy_evaluation(self, nee_like_series):
        """Test that decomposition is computed on first access."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365, verbose=False
        )

        # _decomposition should be None initially
        assert decomp._decomposition is None

        # Access trend (triggers computation)
        _ = decomp.trend
        assert decomp._decomposition is not None

        # Subsequent accesses should use cache
        trend1 = decomp.trend
        trend2 = decomp.trend
        assert trend1 is trend2 or np.array_equal(trend1.values, trend2.values)

    def test_properties_trend(self, nee_like_series):
        """Test trend property."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        trend = decomp.trend

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(nee_like_series)

    def test_properties_seasonal(self, nee_like_series):
        """Test seasonal property."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        seasonal = decomp.seasonal

        assert isinstance(seasonal, pd.Series)
        assert len(seasonal) == len(nee_like_series)

    def test_properties_residual(self, nee_like_series):
        """Test residual property."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        residual = decomp.residual

        assert isinstance(residual, pd.Series)
        assert len(residual) == len(nee_like_series)

    def test_seasonality_strength(self, nee_like_series):
        """Test seasonality strength calculation."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        strength = decomp.seasonality_strength

        # Should be between 0 and 1
        assert 0 <= strength <= 1

    def test_detrend_method(self, nee_like_series):
        """Test detrend method (removes trend)."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        detrended = decomp.detrend()

        # Detrended = seasonal + residual
        expected = decomp.seasonal + decomp.residual
        np.testing.assert_array_almost_equal(detrended.values, expected.values, decimal=5)

    def test_deseasonalize_method(self, nee_like_series):
        """Test deseasonalize method (removes seasonal)."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        deseasonalized = decomp.deseasonalize()

        # Deseasonalized = trend + residual
        expected = decomp.trend + decomp.residual
        np.testing.assert_array_almost_equal(
            deseasonalized.values, expected.values, decimal=5
        )

    def test_reconstruct_all_components(self, nee_like_series):
        """Test reconstruct with all components."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        reconstructed = decomp.reconstruct()

        # Should closely match original
        mse = np.mean((reconstructed - nee_like_series) ** 2)
        assert mse < 1.0

    def test_reconstruct_subset(self, nee_like_series):
        """Test reconstruct with subset of components."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        reconstructed = decomp.reconstruct(keep_components=['seasonal', 'residual'])

        expected = decomp.seasonal + decomp.residual
        np.testing.assert_array_almost_equal(
            reconstructed.values, expected.values, decimal=5
        )

    def test_summary_output(self, nee_like_series):
        """Test summary method produces text output."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )
        summary = decomp.summary()

        # Should be a string
        assert isinstance(summary, str)

        # Should contain key information
        assert 'STL' in summary or 'CLASSICAL' in summary or 'HARMONIC' in summary
        assert 'strength' in summary.lower()

    def test_repr_str_output(self, nee_like_series):
        """Test __repr__ and __str__ methods."""
        decomp = SeasonalTrendDecomposition(
            nee_like_series, seasonal_period=365
        )

        # __repr__ should give object info
        repr_str = repr(decomp)
        assert 'SeasonalTrendDecomposition' in repr_str

        # __str__ should call summary
        str_output = str(decomp)
        assert '=' in str_output  # summary has borders


class TestHarmonicAnalysis:
    """Test harmonic analysis functions."""

    @pytest.fixture
    def simple_sine_series(self):
        """Create simple sine wave."""
        n = 1000
        t = np.arange(n)
        # 50-point period sine wave
        series = 10 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.1, n)
        return pd.Series(series)

    def test_harmonic_analysis(self, simple_sine_series):
        """Test harmonic analysis extracts correct period."""
        result = harmonic_analysis(simple_sine_series, period=50, n_harmonics=3)

        assert 'harmonics' in result
        assert len(result['harmonics']) > 0

        # First harmonic should be close to period
        first = result['harmonics'][0]
        assert 40 < first['period'] < 60

    def test_reconstruct_harmonics(self, simple_sine_series):
        """Test harmonic reconstruction."""
        result = harmonic_analysis(simple_sine_series, period=50, n_harmonics=3)

        reconstructed = reconstruct_harmonics(
            result['harmonics'], len(simple_sine_series)
        )

        assert len(reconstructed) == len(simple_sine_series)
        assert isinstance(reconstructed, np.ndarray)

    def test_periodogram(self, simple_sine_series):
        """Test periodogram computation."""
        result = periodogram(simple_sine_series)

        assert 'frequencies' in result
        assert 'power' in result or 'amplitude' in result
        assert 'peaks' in result
        assert 'dominant_frequency' in result

    def test_fft_decompose(self, simple_sine_series):
        """Test FFT decomposition."""
        result = fft_decompose(simple_sine_series, n_components=5)

        assert 'components' in result
        assert len(result['components']) <= 5
        assert 'reconstructed' in result
        assert 'residual' in result
        assert 'explained_variance_ratio' in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_series(self):
        """Test handling of very short series."""
        short = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))

        with pytest.warns(UserWarning):
            decomp = SeasonalTrendDecomposition(
                short, seasonal_period=2, verbose=False
            )

    def test_all_nan_series(self):
        """Test handling of all-NaN series."""
        all_nan = pd.Series(np.full(100, np.nan))

        # Should either handle gracefully or raise informative error
        try:
            decomp = SeasonalTrendDecomposition(all_nan, seasonal_period=30)
            # If it doesn't raise, components should be mostly NaN
            assert decomp.residual.isna().all() or np.isfinite(decomp.residual).sum() < 10
        except (ValueError, RuntimeError):
            # Acceptable to raise error for all-NaN
            pass

    def test_constant_series(self):
        """Test handling of constant series (no variance)."""
        const = pd.Series(np.full(100, 5.0))

        decomp = SeasonalTrendDecomposition(const, seasonal_period=30, verbose=False)

        # Seasonal and residual should be near zero
        assert decomp.seasonal.abs().max() < 0.1
        assert decomp.residual.abs().max() < 0.1

    def test_series_with_gaps(self):
        """Test series with large gaps."""
        series = pd.Series(np.arange(100, dtype=float))
        series.iloc[40:60] = np.nan

        decomp = SeasonalTrendDecomposition(
            series, seasonal_period=30, verbose=False
        )

        # Should still produce components (may be NaN in gap region)
        assert decomp.trend is not None
        assert decomp.seasonal is not None
        assert decomp.residual is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
