"""
Examples for seasonal-trend decomposition using SeasonalTrendDecomposition.

Run this script to see decomposition results and analysis:
    python examples/analyses/seasonaltrend.py
"""
import numpy as np
import pandas as pd
import diive as dv


def example_seasonaltrend_harmonic_decomposition():
    """Decompose net ecosystem productivity into trend, seasonal, and residual.

    Demonstrates seasonal-trend decomposition using the fast harmonic (FFT-based)
    method on multi-year flux data. Shows how to access decomposition components,
    analyze seasonality strength, and interpret ecosystem patterns (annual cycles,
    long-term trends, anomalies).
    """
    # Load example data
    print("Loading example NEE data...")
    df = dv.load_exampledata_parquet()

    # Use NEE data (flux data with annual cycles)
    # For faster example, use a 2-year subset and interpolate missing values
    nee_series = df['NEE_CUT_REF_orig'].loc['2015':'2016'].copy()
    nee_series = nee_series.interpolate(method='linear', limit_direction='both').bfill().ffill()

    print(f"Series length: {len(nee_series)} records")
    print(f"Data range: {nee_series.index[0]} to {nee_series.index[-1]}")
    print(f"Valid data after interpolation: {nee_series.notna().sum()}")

    # Create quality flags (assuming all data is good quality for this example)
    quality_flags = pd.Series(
        0.8 * np.ones(len(nee_series)), index=nee_series.index
    )

    print("\nDecomposing with harmonic method (fast FFT-based)...")
    decomp = dv.SeasonalTrendDecomposition(
        nee_series,
        quality=quality_flags,
        method='harmonic',
        n_harmonics=10,
        verbose=True
    )

    print("\nDecomposition results:")
    print(decomp.summary())

    print(f"\nSeasonality strength: {decomp.seasonality_strength:.4f}")
    print(f"Trend range: {decomp.trend.min():.3f} to {decomp.trend.max():.3f}")
    print(f"Seasonal amplitude (std): {decomp.seasonal.std():.3f}")
    print(f"Residual std: {decomp.residual.std():.3f}")

    # Interpretation guide
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)

    seasonality = decomp.seasonality_strength
    if seasonality > 0.5:
        seasonality_text = "STRONG: Data has dominant recurring patterns (e.g., daily/annual cycles)"
    elif seasonality > 0.2:
        seasonality_text = "MODERATE: Seasonal patterns present but not dominant"
    else:
        seasonality_text = "WEAK: Data variability driven mainly by trend and random variation"

    print(f"\nSeasonality Strength ({seasonality:.3f}):")
    print(f"  {seasonality_text}")
    print(f"  Interpretation: {seasonality:.1%} of variance explained by seasonal patterns")

    print(f"\nTrend Component:")
    print(f"  Range: {decomp.trend.min():.3f} to {decomp.trend.max():.3f}")
    print(f"  Interpretation: Long-term direction of the ecosystem (e.g., greening, degradation)")
    print(f"  Use case: Detect climate change impacts or ecosystem recovery trends")

    print(f"\nSeasonal Component (amplitude = {decomp.seasonal.std():.3f}):")
    print(f"  Interpretation: Recurring patterns (daily/weekly/annual cycles)")
    print(f"  For NEE: Reflects photosynthetic rhythm (CO2 uptake during day, release at night)")
    print(f"  Use case: Understand ecosystem diurnal/annual behavior")

    print(f"\nResidual Component (std = {decomp.residual.std():.3f}):")
    print(f"  Interpretation: Noise, anomalies, and unexplained variability")
    print(f"  High residuals: Events not captured by trend/seasonal (droughts, storms, fires)")
    print(f"  Use case: Anomaly detection and quality control")

    print(f"\nSeries Reconstruction:")
    print(f"  Original = Trend + Seasonal + Residual")
    print(f"  Detrended = Seasonal + Residual")
    print(f"  Deseasonalized = Trend + Residual")

    print("=" * 60)


if __name__ == '__main__':
    example_seasonaltrend_harmonic_decomposition()
